//! A field's range over a box and the parts that can matter there (language.md §20.4): a whole
//! power's interval over a range holding 0 (§14.2), a bounded derived gradient where a length is
//! written out or smooth unions fold (§14.2), and a union asked only where its parts can matter
//! (`Surface::live`, `distance_live`): the same distance as the whole ask, to the bit, at many
//! points of many boxes, on the CPU. The GPU's: toolchain/tests/masked.

use wrela_host::{CpuBuild, Host, Value};
use wrela_tests::{one_u32, page};

const SRC: &str = "use std::derive::{Box3, Interval, gradient, interval}
use std::field::{Live, RoundCone, Surface, cuboid, fbm, half_space, round_cone, sphere}
use std::quat::Quat

/// The engine's creature's seventeen parts at rest (toolchain/tests/masked).
fn cones() -> [RoundCone; 17] {
    [
        round_cone(vec3(0.0, 0.5694, -0.0258), vec3(0.0, 0.7866, 0.0042), r_top: 0.102, r_bottom: 0.096),
        round_cone(vec3(0.0, 0.7866, 0.0042), vec3(0.0, 0.912, -0.0246), r_top: 0.096, r_bottom: 0.06),
        round_cone(vec3(0.0, 1.02, 0.0018), vec3(0.0, 1.056, 0.0078), r_top: 0.15, r_bottom: 0.138),
        round_cone(vec3(0.1272, 0.873, -0.039), vec3(0.2778, 0.873, -0.0438), r_top: 0.039, r_bottom: 0.0348),
        round_cone(vec3(0.2778, 0.873, -0.0438), vec3(0.4236, 0.873, -0.039), r_top: 0.0348, r_bottom: 0.03),
        round_cone(vec3(0.4236, 0.873, -0.039), vec3(0.4932, 0.8706, -0.03), r_top: 0.03, r_bottom: 0.027),
        round_cone(vec3(0.0684, 0.5826, -0.0216), vec3(0.0684, 0.3252, -0.0216), r_top: 0.054, r_bottom: 0.0432),
        round_cone(vec3(0.0684, 0.3252, -0.0216), vec3(0.0684, 0.0516, -0.0528), r_top: 0.0432, r_bottom: 0.0336),
        round_cone(vec3(0.0684, 0.0516, -0.0528), vec3(0.0684, 0.009, 0.033), r_top: 0.0336, r_bottom: 0.0276),
        round_cone(vec3(0.078, 1.14, 0.0018), vec3(0.12, 1.272, -0.0102), r_top: 0.03, r_bottom: 0.0075),
        round_cone(vec3(-0.1272, 0.873, -0.039), vec3(-0.2778, 0.873, -0.0438), r_top: 0.039, r_bottom: 0.0348),
        round_cone(vec3(-0.2778, 0.873, -0.0438), vec3(-0.4236, 0.873, -0.039), r_top: 0.0348, r_bottom: 0.03),
        round_cone(vec3(-0.4236, 0.873, -0.039), vec3(-0.4932, 0.8706, -0.03), r_top: 0.03, r_bottom: 0.027),
        round_cone(vec3(-0.0684, 0.5826, -0.0216), vec3(-0.0684, 0.3252, -0.0216), r_top: 0.054, r_bottom: 0.0432),
        round_cone(vec3(-0.0684, 0.3252, -0.0216), vec3(-0.0684, 0.0516, -0.0528), r_top: 0.0432, r_bottom: 0.0336),
        round_cone(vec3(-0.0684, 0.0516, -0.0528), vec3(-0.0684, 0.009, 0.033), r_top: 0.0336, r_bottom: 0.0276),
        round_cone(vec3(-0.078, 1.14, 0.0018), vec3(-0.12, 1.272, -0.0102), r_top: 0.03, r_bottom: 0.0075),
    ]
}

/// Box `i` of side `side`, its centre spread through the creature's room.
fn box_at(i: u32, side: f32) -> Box3 {
    let c = Box3 { lo: vec3(-0.6, -0.1, -0.4), hi: vec3(0.6, 1.4, 0.4) }.sample(17, i)
    Box3 { lo: c - vec3(0.5 * side), hi: c + vec3(0.5 * side) }
}

pub fn pow_range(lo: f32, hi: f32, y: f32) -> vec2 {
    let r = interval(|x: f32| pow(x, y), Interval { lo, hi })
    vec2(r.lo, r.hi)
}

/// Over `n` boxes of side `side`: the most any box's gradient range reaches (its longest
/// corner), and how many of 32 sampled gradients a box fall outside their box's range.
fn steep<S: Surface>(s: S, side: f32, n: u32) -> vec2 {
    var most = 0.0
    var misses: u32 = 0
    for i in 0..n {
        let b = box_at(i, side)
        let g = s.range(b).gradient
        most = max(most, length(max(abs(g.lo), abs(g.hi))))
        for k in 0..32 {
            let q = b.sample(5, k)
            let d = s.gradient(q)
            if d.x < g.lo.x || d.y < g.lo.y || d.z < g.lo.z || d.x > g.hi.x || d.y > g.hi.y || d.z > g.hi.z {
                misses += 1
            }
        }
    }
    vec2(most, f32(misses))
}

/// `steep` of one round cone (kind 0, its axis in many boxes), the seventeen joined smoothly
/// (1), or joined sharply (2).
pub fn steepest(kind: u32, side: f32, n: u32) -> vec2 {
    let c = cones()
    if kind == 0 {
        steep(c[0], side, n)
    } else if kind == 1 {
        steep(c.smooth_union(k: 0.03), side, n)
    } else {
        steep(c.union(), side, n)
    }
}

/// Over `n` boxes of side `side`, 64 points each: how many masked asks differ from the whole
/// ask in any bit, and the parts live a box, summed.
fn masked<S: Surface>(s: S, side: f32, n: u32) -> vec2 {
    var differ: u32 = 0
    var live: u32 = 0
    for i in 0..n {
        let b = box_at(i, side)
        let l = s.live(b)
        live += l.count(s.parts())
        for k in 0..64 {
            let q = b.sample(9, k)
            if bitcast_u32(s.distance(q)) != bitcast_u32(s.distance_live(q, l)) {
                differ += 1
            }
        }
    }
    vec2(f32(differ), f32(live))
}

/// `masked` of: the creature with its wand (0); the seventeen joined sharply (1); and a tree
/// of every union std has under its moves and channels (2).
pub fn masks(kind: u32, side: f32, n: u32) -> vec2 {
    let c = cones()
    let wand = round_cone(vec3(-0.4236, 0.873, -0.039), vec3(-0.4956, 1.053, 0.261), r_top: 0.00504, r_bottom: 0.00504)
    if kind == 0 {
        masked(c.smooth_union(k: 0.03).with(vec3(0.4)).smooth_union(wand.with(vec3(0.2)), k: 0.004), side, n)
    } else if kind == 1 {
        masked(c.union(), side, n)
    } else {
        let turn = Quat::axis_angle(normalize(vec3(1.0, 2.0, 0.5)), 0.7)
        let legs = [c[6], c[7], c[8], c[13], c[14], c[15]].smooth_union(k: 0.02).translate(vec3(0.0, 0.1, 0.0))
        let arms = [c[3], c[4], c[5], c[10], c[11], c[12]].union().rotate(turn).scale(1.3)
        let head = sphere(0.15).translate(vec3(0.0, 1.05, 0.0)).smooth_union(c[9], k: 0.05).union(c[16])
        let block = cuboid(vec3(0.2, 0.1, 0.1)).translate(vec3(0.3, 0.2, 0.0)).displace(fbm(4.0, 3), amp: 0.01)
        let body = c[0].smooth_union(c[1], k: 0.04).with(vec3(1.0)).union(head)
        masked(legs.union(arms).smooth_union(body, k: 0.03).union(block).to_bound(0.5), side, n)
    }
}

/// Over `n` rays from all round the surface's room (some starting inside it), 512 points each
/// along the first 8 m: how many points inside the surface fall outside its `meets` (by more
/// than 0.1 mm); of the rays `meets` calls exact, how many have a point inside before `near`, or
/// a distance at `near` off 0 by more than 0.1 mm (or, at 0, not inside); how many it calls
/// exact; and the worst distance at an exact `near`.
fn meet_check<S: Surface>(s: S, n: u32) -> vec4 {
    var outside: u32 = 0
    var wrong: u32 = 0
    var exact: u32 = 0
    var worst = 0.0
    for i in 0..n {
        let o = Box3 { lo: vec3(-3.0, -1.0, -3.0), hi: vec3(3.0, 3.0, 3.0) }.sample(23, i)
        let aim = Box3 { lo: vec3(-1.0, -0.2, -1.0), hi: vec3(1.0, 1.5, 1.0) }.sample(29, i)
        let d = normalize(aim - o)
        let m = s.meets(o, d)
        var first = -1.0
        for k in 0..512 {
            let t = f32(k) * (8.0 / 512.0)
            if s.distance(o + d * t) <= 0.0 {
                if m.is_empty() || t < m.near - 0.0001 || t > m.far + 0.0001 {
                    outside += 1
                }
                if first < 0.0 {
                    first = t
                }
            }
        }
        if m.exact() {
            exact += 1
            if m.is_empty() {
                if first >= 0.0 {
                    wrong += 1
                }
            } else {
                let f = s.distance(o + d * m.near)
                let at = abs(f) < 0.0001 || (m.near == 0.0 && f <= 0.0)
                if !at || (first >= 0.0 && first < m.near - 0.0001) {
                    wrong += 1
                }
                worst = max(worst, if m.near == 0.0 { 0.0 } else { abs(f) })
            }
        }
    }
    vec4(f32(outside), f32(wrong), f32(exact), worst)
}

/// `meet_check` of: a ball (0), a box turned and scaled (1), a round cone (2), the engine's stage
/// (two boxes and a ball, joined sharply: 3), the creature (4), the tree of every union (5), a
/// ball cut by a plane (6), a displaced box (7), a mirrored cone (8).
pub fn meet_checks(kind: u32, n: u32) -> vec4 {
    let c = cones()
    let turn = Quat::axis_angle(normalize(vec3(1.0, 2.0, 0.5)), 0.7)
    if kind == 0 {
        meet_check(sphere(0.7).translate(vec3(0.2, 0.5, -0.1)), n)
    } else if kind == 1 {
        meet_check(cuboid(vec3(0.5, 0.2, 0.3)).rotate(turn).scale(1.3).translate(vec3(0.0, 0.4, 0.0)), n)
    } else if kind == 2 {
        meet_check(round_cone(vec3(-0.3, 0.2, 0.0), vec3(0.4, 0.9, 0.2), r_top: 0.3, r_bottom: 0.1), n)
    } else if kind == 3 {
        let ground = cuboid(vec3(8.0, 0.25, 8.0)).translate(vec3(0.0, -0.25, 0.0)).with(vec3(0.3))
        let wall = cuboid(vec3(3.0, 1.5, 0.1)).translate(vec3(0.0, 1.5, 0.0)).with(vec3(0.4))
        let ball = sphere(1.0).translate(vec3(-0.5, 1.0, 0.6)).with(vec3(0.5))
        meet_check(ground.union(wall).union(ball), n)
    } else if kind == 4 {
        let wand = round_cone(vec3(-0.4236, 0.873, -0.039), vec3(-0.4956, 1.053, 0.261), r_top: 0.00504, r_bottom: 0.00504)
        meet_check(c.smooth_union(k: 0.03).with(vec3(0.4)).smooth_union(wand.with(vec3(0.2)), k: 0.004), n)
    } else if kind == 5 {
        let legs = [c[6], c[7], c[8], c[13], c[14], c[15]].smooth_union(k: 0.02).translate(vec3(0.0, 0.1, 0.0))
        let arms = [c[3], c[4], c[5], c[10], c[11], c[12]].union().rotate(turn).scale(1.3)
        let head = sphere(0.15).translate(vec3(0.0, 1.05, 0.0)).smooth_union(c[9], k: 0.05).union(c[16])
        let body = c[0].smooth_union(c[1], k: 0.04).with(vec3(1.0)).union(head)
        meet_check(legs.union(arms).smooth_union(body, k: 0.03), n)
    } else if kind == 6 {
        meet_check(sphere(0.8).translate(vec3(0.0, 0.5, 0.0)).intersect(half_space(vec3(0.3, 1.0, 0.2), 0.6)), n)
    } else if kind == 7 {
        meet_check(cuboid(vec3(0.4, 0.3, 0.5)).translate(vec3(0.1, 0.4, 0.0)).displace(fbm(4.0, 3), amp: 0.05), n)
    } else {
        meet_check(round_cone(vec3(0.2, 0.2, 0.0), vec3(0.6, 0.9, 0.2), r_top: 0.2, r_bottom: 0.1).mirror_x(), n)
    }
}
";

/// The package, built once for the three tests.
fn host() -> wrela_host::CpuHost {
    static BUILT: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    let build = BUILT.get_or_init(|| {
        let dir = crate::package("ranges", SRC);
        wrela_tests::must_build(&dir, &dir.join("build"));
        dir.join("build")
    });
    CpuBuild::load(build).expect("load").start().expect("start")
}

fn two(host: &mut wrela_host::CpuHost, name: &str, args: &[Value]) -> (f64, f64) {
    match host.call_export(name, args).expect(name)[..] {
        [Value::F32(a), Value::F32(b)] => (f64::from(a), f64::from(b)),
        ref other => panic!("{name} returned {other:?}"),
    }
}

/// `pow(x, y)` of a whole `y` over a range holding negatives is the power, as std's CPU `pow`
/// computes it (a negative base and a whole exponent: `pown`), not every number.
#[test]
fn a_whole_power_of_a_range_holding_0_is_bounded() {
    let mut host = host();
    for (lo, hi, y) in [
        (-2.0f32, 3.0f32, 2.0f32),
        (-2.0, 3.0, 3.0),
        (-3.0, -1.0, 2.0),
        (-3.0, -1.0, 3.0),
        (-3.0, -1.0, -2.0),
        (0.5, 2.0, 4.0),
        (-0.5, 0.25, 5.0),
    ] {
        let (rl, rh) = two(&mut host, "pow_range", &[lo, hi, y].map(Value::F32));
        let (mut sl, mut sh) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..=1000 {
            let x = lo + (hi - lo) * i as f32 / 1000.0;
            let p = f64::from(x.powi(y as i32));
            sl = sl.min(p);
            sh = sh.max(p);
        }
        let e = 1e-5 * (1.0 + sl.abs().max(sh.abs()));
        assert!(
            rl <= sl && rh >= sh,
            "pow over [{lo}, {hi}] to {y}: [{rl}, {rh}] misses [{sl}, {sh}]"
        );
        assert!(
            rl > sl - e && rh < sh + e,
            "pow over [{lo}, {hi}] to {y}: [{rl}, {rh}] loose of [{sl}, {sh}]"
        );
    }
    // A negative power of a range holding 0 has a pole there: every number.
    let (rl, rh) = two(&mut host, "pow_range", &[-1.0f32, 1.0, -2.0].map(Value::F32));
    assert!(rl == f64::NEG_INFINITY && rh == f64::INFINITY, "[{rl}, {rh}]");
}

/// A round cone's derived gradient is bounded over boxes holding its axis (`sqrt(x² + z²)`
/// written out is differentiated, for its interval, as a length is), and so is a fold of
/// seventeen smooth unions (each `smin`'s derivative a mix of its operands', for its interval):
/// the corner of the box of a cone's gradients within 2 (a unit vector's is within √3), and the corner of the box of gradients
/// within its parts' for the folds, at every box size. Every sampled gradient is inside.
#[test]
fn a_length_written_out_and_a_fold_of_smooth_unions_have_bounded_gradients() {
    let mut host = host();
    for side in [2.0f32, 0.5, 0.125, 0.03, 0.008] {
        for (kind, most) in [(0, 2.0), (1, 3.2), (2, 3.2)] {
            let args = [Value::I32(kind), Value::F32(side), Value::I32(200)];
            let (steepest, misses) = two(&mut host, "steepest", &args);
            assert_eq!(misses, 0.0, "kind {kind}, boxes of {side} m: sampled gradients outside");
            assert!(steepest <= most, "kind {kind}, boxes of {side} m: gradient up to {steepest}");
        }
    }
}

/// A masked ask (`distance_live` with what `live` found for a box) is the whole ask, to the
/// bit, at every point of every box: the creature (a fold of seventeen smooth unions joined to
/// a wand), the seventeen joined sharply, and a tree of every union std has under moves,
/// turns, a scale, channels, a displacement and `to_bound`. In small boxes few parts are live.
#[test]
fn a_masked_ask_is_the_whole_ask_to_the_bit() {
    let mut host = host();
    for (kind, parts) in [(0, 18.0), (1, 17.0), (2, 18.0)] {
        for side in [2.0f32, 0.5, 0.125, 0.03, 0.008] {
            let args = [Value::I32(kind), Value::F32(side), Value::I32(300)];
            let (differ, live) = two(&mut host, "masks", &args);
            assert_eq!(differ, 0.0, "kind {kind}, boxes of {side} m: masked asks differ");
            let mean = live / 300.0;
            assert!(mean >= 1.0 && mean <= parts, "kind {kind}: {mean} live of {parts}");
            if side <= 0.03 {
                assert!(
                    mean < 0.25 * parts,
                    "kind {kind}, boxes of {side} m: {mean} live of {parts}"
                );
            }
        }
    }
}

/// On the GPU, the masked fixture's two pictures (the creature asked whole, and asked only by
/// each brick's live parts) are the same to the bit, and a kernel finds the CPU's masks.
#[test]
#[ignore = "needs a GPU"]
fn masked_and_whole_draw_the_same_picture() {
    let (dir, _) = page("toolchain/tests/masked", "masked-native");
    let mut host = Host::load(&dir).expect("load");
    let times = crate::frame_times(6);
    host.run_frames(&times, 320, 180).expect("run");
    assert_eq!(one_u32(&mut host, "differing", &[]), 0, "pixels differ");
    assert_eq!(
        one_u32(&mut host, "kernel_misses", &[]),
        0,
        "a kernel's masks leave out live parts"
    );
    assert_eq!(
        one_u32(&mut host, "counts_differing", &[]),
        0,
        "a kernel's counts of what an ask costs differ from the CPU's"
    );
}

/// Where a ray can meet a surface (`Surface::meets`, §20.5): no point inside a surface lies
/// outside the stretch, for std's shapes, the engine's stage and creature, and every combinator;
/// where it says exact (std's shapes and their sharp unions and intersections), `near` is the
/// first point inside, on the surface to 0.1 mm.
#[test]
fn a_ray_meets_a_surface_only_where_meets_says() {
    let mut host = host();
    let mut exact = Vec::new();
    for kind in 0..9u32 {
        let r = host
            .call_export("meet_checks", &[Value::I32(kind as i32), Value::I32(2000)])
            .expect("meet_checks");
        let [Value::F32(outside), Value::F32(wrong), Value::F32(n), Value::F32(worst)] = r[..]
        else {
            panic!("{r:?}")
        };
        assert_eq!(outside, 0.0, "kind {kind}: points inside outside the stretch");
        assert_eq!(
            wrong, 0.0,
            "kind {kind}: exact rays whose near isn't the first point inside (worst {worst})"
        );
        exact.push(n);
    }
    // std's shapes, the stage and a cut by a plane are exact; the smooth, displaced and mirrored aren't.
    for kind in [0, 1, 2, 3, 6] {
        assert_eq!(exact[kind], 2000.0, "kind {kind} is exact: {exact:?}");
    }
    for kind in [4, 5, 7, 8] {
        assert!(exact[kind] < 2000.0, "kind {kind} isn't exact everywhere: {exact:?}");
    }
}

/// On the GPU, the meets fixture's two ways of finding each pixel's hit (marched from the bound,
/// and from `meets`: the stage's exactly, the creature's from a stretch) draw the same pictures:
/// pixels met by one way only, or shaded by their facing more than 1/255 apart, under 0.1%.
#[test]
#[ignore = "needs a GPU"]
fn meeting_and_marching_draw_the_same_picture() {
    let (dir, _) = page("toolchain/tests/meets", "meets-native");
    let mut host = Host::load(&dir).expect("load");
    let times = crate::frame_times(6);
    host.run_frames(&times, 320, 180).expect("run");
    assert_eq!(one_u32(&mut host, "compared_scenes", &[]), 2, "both scenes compared");
    for scene in 0..2 {
        let arg = [Value::I32(scene)];
        let differ = one_u32(&mut host, "missed", &arg) + one_u32(&mut host, "shaded", &arg);
        assert!(differ <= 57, "scene {scene}: {differ} of 57,600 pixels differ");
    }
}
