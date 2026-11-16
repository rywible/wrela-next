//! Every rule, checked on the GPU against the CPU: a wrela program whose kernel computes each
//! rule's two sides on inputs it makes from a hash (within the rule's facts), writes the inputs
//! it used and both results, and an f64 evaluation of the left side on the CPU as the reference.
//!
//! The error is measured in ulps of the **scale**: the largest magnitude of any value the two
//! sides compute on the way (their operands included), as an f64 evaluation gives them. A sum's
//! rounding is an ulp of its largest term, not of its result: `(a + b) - b` against `a` is wrong
//! by ulps of `b`. The bounds ([`bound`]) are stated per group.

use crate::rules::{Cond, Group, SPECS, expand};
use egg::{ENodeOrVar, Language, Pattern, PatternAst, Var};
use std::collections::HashMap;
use std::fmt::Write;
use wrela_ir::Builtin;

use crate::lang::{Node, Uop};

/// The inputs each case and sample get.
pub const SAMPLES: usize = 256;
/// Floats in a case's record: up to three vec3 inputs, then the left side's three components,
/// then the right's.
pub const ROW: usize = 15;

/// The most a rule's two sides may differ on the GPU, in ulps of the scale.
pub fn bound(g: Group) -> f64 {
    match g {
        Group::Exact | Group::Interval => 4.0,
        Group::Approx | Group::Reassoc => 64.0,
    }
}

pub struct Case {
    pub name: String,
    pub group: Group,
    pub lhs: PatternAst<Node>,
    pub rhs: PatternAst<Node>,
    /// Each variable and its width.
    pub vars: Vec<(Var, u8)>,
    pub conds: &'static [Cond],
}

fn vector_only(s: &str) -> bool {
    ["length", "dot", "normalize", "distance"].iter().any(|k| s.contains(k))
}

/// The cases: each rule at width 1 and 3 (a vector-only rule at 3, a comparison at 1), and the
/// splat family at width 3.
pub fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for spec in SPECS {
        for (name, lhs, rhs, w) in expand(spec) {
            let widths: Vec<u8> = match w {
                0 if vector_only(spec.lhs) || vector_only(spec.rhs) => vec![3],
                0 if spec.name.starts_with("lt-") => vec![1],
                0 => vec![1, 3],
                1 | 3 => vec![w],
                _ => vec![],
            };
            for width in widths {
                let lhs_p: Pattern<Node> = lhs.parse().expect("parses");
                let rhs_p: Pattern<Node> = rhs.parse().expect("parses");
                let vars = lhs_p
                    .vars()
                    .into_iter()
                    .map(|v| {
                        let scalar = splatted(&lhs_p.ast).contains(&v);
                        (v, if scalar { 1 } else { width })
                    })
                    .collect();
                let name = if w == 0 { format!("{name}@{width}") } else { name.clone() };
                out.push(Case {
                    name,
                    group: spec.group,
                    lhs: lhs_p.ast,
                    rhs: rhs_p.ast,
                    vars,
                    conds: spec.conds,
                });
            }
        }
    }
    for op in ["+", "-", "*", "/", "min", "max"] {
        let lhs: Pattern<Node> = format!("({op} (splat3 ?a) (splat3 ?b))").parse().expect("parses");
        let rhs: Pattern<Node> = format!("(splat3 ({op} ?a ?b))").parse().expect("parses");
        let vars = lhs.vars().into_iter().map(|v| (v, 1)).collect();
        out.push(Case {
            name: format!("splat-op{op}/3"),
            group: Group::Exact,
            lhs: lhs.ast,
            rhs: rhs.ast,
            vars,
            conds: &[],
        });
    }
    out
}

// ---- the program -------------------------------------------------------------------------------

fn wrela_expr(ast: &PatternAst<Node>, i: usize, names: &HashMap<Var, String>) -> String {
    let n = &ast.as_ref()[i];
    let node = match n {
        ENodeOrVar::Var(v) => return names[v].clone(),
        ENodeOrVar::ENode(e) => e,
    };
    let k: Vec<String> =
        node.children().iter().map(|&c| wrela_expr(ast, usize::from(c), names)).collect();
    match node {
        Node::Lit(l) => match l.as_f32() {
            Some(x) if x < 0.0 => format!("({x:?})"),
            Some(x) => format!("{x:?}"),
            None => {
                if l.bits != 0 {
                    "true".into()
                } else {
                    "false".into()
                }
            }
        },
        Node::Bin(..) => format!("({} {node} {})", k[0], k[1]),
        Node::Un(Uop::Neg, _) => format!("(-{})", k[0]),
        Node::Call(b, _) => {
            let name = match b {
                Builtin::InverseSqrt => "inverse_sqrt".to_string(),
                _ => node.to_string(),
            };
            format!("{name}({})", k.join(", "))
        }
        Node::Splat(n, _) => format!("vec{n}({})", k[0]),
        Node::Extract(c, _) => format!("{}.{}", k[0], ["x", "y", "z", "w"][*c as usize]),
        Node::Select(_) => format!("select(if_false: {}, if_true: {}, cond: {})", k[2], k[1], k[0]),
        _ => panic!("no wrela for {node:?}"),
    }
}

fn width_of(ast: &PatternAst<Node>, vars: &[(Var, u8)]) -> usize {
    let mut x = HashMap::new();
    for (v, w) in vars {
        x.insert(*v, vec![1.0; usize::from(*w)]);
    }
    eval(ast, &x).0.len()
}

/// The program: its `main.wrela`.
pub fn source(cases: &[Case]) -> String {
    let mut s = String::new();
    let total = cases.len() * SAMPLES;
    writeln!(s, "// The e-graph spike's rules, both sides on the GPU (toolchain/egraph/src/test_program.rs).").unwrap();
    writeln!(s, "use std::gpu::{{GlobalId, GpuBuffer, Slots, buffer, dispatch}}").unwrap();
    writeln!(s, "use std::hash::pcg\n").unwrap();
    writeln!(s, "struct Row: Copy + GpuData {{\n    w: [f32; {ROW}],\n}}\n").unwrap();
    writeln!(s, "/// In [0, 1).\nfn u01(seed: u32) -> f32 {{\n    f32(pcg(seed) >> 8) * (1.0 / 16777216.0)\n}}\n").unwrap();
    writeln!(s, "/// A sign and a magnitude from 1/128 to 128.\nfn val(seed: u32) -> f32 {{\n    let m = exp2(u01(seed * 2 + 1) * 14.0 - 7.0)\n    if u01(seed * 2) < 0.5 {{ -m }} else {{ m }}\n}}\n").unwrap();
    writeln!(s, "@compute(64)\nfn rules(out: mut Slots<Row>, id: GlobalId) {{").unwrap();
    writeln!(s, "    let case = id.x / {SAMPLES}\n    let seed = id.x * 16 + 7").unwrap();
    writeln!(s, "    var w: [f32; {ROW}] = [0.0; {ROW}]").unwrap();
    for (ci, c) in cases.iter().enumerate() {
        writeln!(s, "    if case == {ci} {{").unwrap();
        let mut names = HashMap::new();
        let conds = selected(&c.lhs);
        for (k, (v, width)) in c.vars.iter().enumerate() {
            let name = v.to_string().trim_start_matches('?').to_string();
            let comps: Vec<String> =
                (0..*width).map(|j| format!("val(seed + {})", k * 3 + usize::from(j))).collect();
            let init = if *width == 1 {
                comps[0].clone()
            } else {
                format!("vec{width}({})", comps.join(", "))
            };
            // A select's condition is a bool.
            let init = if conds.contains(v) { format!("{} < 0.0", comps[0]) } else { init };
            writeln!(s, "        var {name} = {init}").unwrap();
            names.insert(*v, name);
        }
        let n = |x: &str| {
            names
                .iter()
                .find(|(v, _)| v.to_string() == format!("?{x}"))
                .map(|(_, s)| s.clone())
                .unwrap_or_else(|| x.to_string())
        };
        let unit = |name: &str, k: usize, width: u8| {
            if width == 1 {
                format!("u01(seed + {})", 40 + k)
            } else {
                format!(
                    "vec{width}(u01(seed + {}), u01(seed + {}), u01(seed + {}))",
                    40 + k,
                    41 + k,
                    42 + k
                )
            }
            .replace("NAME", name)
        };
        for (k, c) in c.conds.iter().enumerate() {
            let w = |x: &str| c_width(&names, &cases[ci].vars, x);
            match *c {
                Cond::NonNeg(a) => writeln!(s, "        {} = abs({})", n(a), n(a)).unwrap(),
                Cond::Pos(a) => writeln!(s, "        {} = abs({}) + 0.001", n(a), n(a)).unwrap(),
                Cond::NonPos(a) => writeln!(s, "        {} = -abs({})", n(a), n(a)).unwrap(),
                Cond::Unit(a) => {
                    writeln!(s, "        {} = {}", n(a), unit(a, k * 3, w(a))).unwrap()
                }
                Cond::Ge(a, b) => {
                    writeln!(s, "        {} = {} + abs({})", n(a), n(b), n(a)).unwrap()
                }
                Cond::Lt(a, b) | Cond::Le(a, b) => {
                    writeln!(s, "        {} = {} - abs({}) - 0.001", n(a), n(b), n(a)).unwrap()
                }
                Cond::In(x, l, h) => {
                    writeln!(s, "        {} = -abs({})", n(l), n(l)).unwrap();
                    writeln!(s, "        {} = abs({})", n(h), n(h)).unwrap();
                    writeln!(
                        s,
                        "        {} = {} + ({} - {}) * {}",
                        n(x),
                        n(l),
                        n(h),
                        n(l),
                        unit(x, k * 3, w(x))
                    )
                    .unwrap();
                }
                Cond::Clean(_) | Cond::Bounded(_) | Cond::Slower(..) => {}
            }
        }
        let mut slot = 0;
        for (v, width) in &c.vars {
            if conds.contains(v) {
                writeln!(
                    s,
                    "        w[{slot}] = select(if_false: 0.0, if_true: 1.0, cond: {})",
                    names[v]
                )
                .unwrap();
                slot += 3;
                continue;
            }
            for j in 0..*width {
                let comp = if *width == 1 {
                    names[v].clone()
                } else {
                    format!("{}.{}", names[v], ["x", "y", "z", "w"][usize::from(j)])
                };
                writeln!(s, "        w[{slot}] = {comp}").unwrap();
                slot += 1;
            }
            slot += 3 - usize::from(*width);
        }
        let out_width = width_of(&c.lhs, &c.vars);
        let boolean = c.name.starts_with("lt-");
        for (side, ast) in [(9, &c.lhs), (12, &c.rhs)] {
            let e = wrela_expr(ast, ast.as_ref().len() - 1, &names);
            let e =
                if boolean { format!("select(if_false: 0.0, if_true: 1.0, cond: {e})") } else { e };
            writeln!(s, "        let r{side} = {e}").unwrap();
            if out_width == 1 {
                writeln!(s, "        w[{side}] = r{side}").unwrap();
            } else {
                for j in 0..out_width {
                    writeln!(s, "        w[{}] = r{side}.{}", side + j, ["x", "y", "z", "w"][j])
                        .unwrap();
                }
            }
        }
        writeln!(s, "    }}").unwrap();
    }
    writeln!(s, "    out[id] = Row {{ w: w }}\n}}\n").unwrap();
    writeln!(s, "pub struct Game {{\n    out: GpuBuffer<Row>,\n    done: bool,\n}}\n").unwrap();
    writeln!(s, "pub fn init() -> Game {{\n    Game {{ out: buffer({total}), done: false }}\n}}\n")
        .unwrap();
    writeln!(
        s,
        "pub fn frame(game: mut Game, time: f32, width: u32, height: u32) {{\n    if game.done {{\n        return\n    }}\n    dispatch(rules.bind(out: mut game.out), groups: {})\n    game.done = true\n}}",
        total.div_ceil(64)
    )
    .unwrap();
    s
}

/// The variables a pattern splats: scalars.
fn splatted(ast: &PatternAst<Node>) -> Vec<Var> {
    let mut out = Vec::new();
    for n in ast.as_ref() {
        if let ENodeOrVar::ENode(Node::Splat(_, [c])) = n
            && let ENodeOrVar::Var(v) = &ast.as_ref()[usize::from(*c)]
        {
            out.push(*v);
        }
    }
    out
}

/// The variables a pattern uses as a select's condition.
fn selected(ast: &PatternAst<Node>) -> Vec<Var> {
    let mut out = Vec::new();
    for n in ast.as_ref() {
        if let ENodeOrVar::ENode(Node::Select([c, _, _])) = n
            && let ENodeOrVar::Var(v) = &ast.as_ref()[usize::from(*c)]
        {
            out.push(*v);
        }
    }
    out
}

fn c_width(names: &HashMap<Var, String>, vars: &[(Var, u8)], x: &str) -> u8 {
    let _ = names;
    vars.iter().find(|(v, _)| v.to_string() == format!("?{x}")).map_or(1, |(_, w)| *w)
}

// ---- the reference -----------------------------------------------------------------------------

/// A pattern's value in f64 given its variables', and the largest magnitude of anything it
/// computed on the way (its operands included).
pub fn eval(ast: &PatternAst<Node>, vars: &HashMap<Var, Vec<f64>>) -> (Vec<f64>, f64) {
    let mut vals: Vec<Vec<f64>> = Vec::new();
    let mut scale: f64 = 0.0;
    for n in ast.as_ref() {
        let v = match n {
            ENodeOrVar::Var(v) => vars[v].clone(),
            ENodeOrVar::ENode(e) => {
                let k: Vec<&Vec<f64>> =
                    e.children().iter().map(|&c| &vals[usize::from(c)]).collect();
                node_value(e, &k)
            }
        };
        for x in &v {
            if x.is_finite() {
                scale = scale.max(x.abs());
            }
        }
        vals.push(v);
    }
    (vals.pop().expect("a pattern has a root"), scale)
}

fn lanes(k: &[&Vec<f64>]) -> usize {
    k.iter().map(|v| v.len()).max().unwrap_or(1)
}

fn at(v: &[f64], i: usize) -> f64 {
    if v.len() == 1 { v[0] } else { v[i] }
}

fn node_value(e: &Node, k: &[&Vec<f64>]) -> Vec<f64> {
    let n = lanes(k);
    let each = |f: &dyn Fn(&[f64]) -> f64| -> Vec<f64> {
        (0..n).map(|i| f(&k.iter().map(|v| at(v, i)).collect::<Vec<_>>())).collect()
    };
    let b2f = |b: bool| if b { 1.0 } else { 0.0 };
    match e {
        Node::Lit(l) => vec![l.as_f32().map_or(l.bits as f64, f64::from)],
        Node::Bin(op, _) => {
            use wrela_ir::BinOp as B;
            match op.op() {
                B::Add => each(&|x| x[0] + x[1]),
                B::Sub => each(&|x| x[0] - x[1]),
                B::Mul => each(&|x| x[0] * x[1]),
                B::Div => each(&|x| x[0] / x[1]),
                B::Lt => each(&|x| b2f(x[0] < x[1])),
                B::Le => each(&|x| b2f(x[0] <= x[1])),
                B::Gt => each(&|x| b2f(x[0] > x[1])),
                B::Ge => each(&|x| b2f(x[0] >= x[1])),
                o => panic!("no reference for {o:?}"),
            }
        }
        Node::Un(Uop::Neg, _) => each(&|x| -x[0]),
        Node::Splat(w, _) => vec![k[0][0]; usize::from(*w)],
        Node::Extract(c, _) => vec![k[0][*c as usize]],
        Node::Select(_) => {
            let n = k[1].len().max(k[2].len());
            (0..n).map(|i| if at(k[0], i) != 0.0 { at(k[1], i) } else { at(k[2], i) }).collect()
        }
        Node::Call(b, _) => {
            let dot = |a: &[f64], c: &[f64]| a.iter().zip(c).map(|(x, y)| x * y).sum::<f64>();
            match b {
                Builtin::Sqrt => each(&|x| x[0].sqrt()),
                Builtin::InverseSqrt => each(&|x| 1.0 / x[0].sqrt()),
                Builtin::Abs => each(&|x| x[0].abs()),
                Builtin::Min => each(&|x| x[0].min(x[1])),
                Builtin::Max => each(&|x| x[0].max(x[1])),
                Builtin::Clamp => each(&|x| x[0].max(x[1]).min(x[2])),
                Builtin::Saturate => each(&|x| x[0].clamp(0.0, 1.0)),
                Builtin::Mix => each(&|x| x[0] + (x[1] - x[0]) * x[2]),
                Builtin::Exp => each(&|x| x[0].exp()),
                Builtin::Exp2 => each(&|x| x[0].exp2()),
                Builtin::Log => each(&|x| x[0].ln()),
                Builtin::Log2 => each(&|x| x[0].log2()),
                Builtin::Pow => each(&|x| x[0].powf(x[1])),
                Builtin::Floor => each(&|x| x[0].floor()),
                Builtin::Sign => each(&|x| {
                    if x[0] > 0.0 {
                        1.0
                    } else if x[0] < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                }),
                Builtin::Step => each(&|x| b2f(x[1] >= x[0])),
                Builtin::Length => vec![dot(k[0], k[0]).sqrt()],
                Builtin::Distance => {
                    let d: Vec<f64> = k[0].iter().zip(k[1].iter()).map(|(x, y)| x - y).collect();
                    vec![dot(&d, &d).sqrt()]
                }
                Builtin::Dot => vec![dot(k[0], k[1])],
                Builtin::Normalize => {
                    let l = dot(k[0], k[0]).sqrt();
                    k[0].iter().map(|x| x / l).collect()
                }
                b => panic!("no reference for {b:?}"),
            }
        }
        n => panic!("no reference for {n:?}"),
    }
}

/// One case's result on the GPU.
#[derive(Debug, Clone)]
pub struct Checked {
    pub name: String,
    pub group: Group,
    /// The most the right side differs from the left, in ulps of the scale, over the samples.
    pub sides: f64,
    /// The most each side differs from the reference, in ulps of the scale.
    pub lhs_error: f64,
    pub rhs_error: f64,
    /// Samples left out: the reference wasn't a finite number (a NaN WGSL leaves indeterminate).
    pub skipped: usize,
    pub passed: bool,
}

/// Checks the kernel's rows (`SAMPLES` per case, in order).
pub fn check(cases: &[Case], rows: &[f32]) -> Vec<Checked> {
    let mut out = Vec::new();
    for (ci, c) in cases.iter().enumerate() {
        let (mut sides, mut le, mut re, mut skipped): (f64, f64, f64, usize) = (0.0, 0.0, 0.0, 0);
        for sample in 0..SAMPLES {
            let row = &rows[(ci * SAMPLES + sample) * ROW..][..ROW];
            let mut vars = HashMap::new();
            let conds = selected(&c.lhs);
            for (k, (v, w)) in c.vars.iter().enumerate() {
                let w = if conds.contains(v) { 1 } else { usize::from(*w) };
                vars.insert(*v, (0..w).map(|j| f64::from(row[k * 3 + j])).collect::<Vec<_>>());
            }
            let (want, s1) = eval(&c.lhs, &vars);
            let (_, s2) = eval(&c.rhs, &vars);
            let ulp = s1.max(s2).max(f64::MIN_POSITIVE) * f64::from(f32::EPSILON);
            if want.iter().any(|x| !x.is_finite()) {
                skipped += 1;
                continue;
            }
            for (j, w) in want.iter().enumerate() {
                let (l, r) = (f64::from(row[9 + j]), f64::from(row[12 + j]));
                let d = |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() { f64::INFINITY } else { (a - b).abs() / ulp }
                };
                sides = sides.max(d(l, r));
                le = le.max(d(l, *w));
                re = re.max(d(r, *w));
            }
        }
        let passed = sides <= bound(c.group) && skipped < SAMPLES;
        out.push(Checked {
            name: c.name.clone(),
            group: c.group,
            sides,
            lhs_error: le,
            rhs_error: re,
            skipped,
            passed,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_case_has_a_reference_and_wrela() {
        let cs = cases();
        assert!(cs.len() > 80, "{}", cs.len());
        let src = source(&cs);
        assert!(src.contains("fn rules"));
    }

    /// The rules on this machine's GPU, through tools/gpu (`cargo test -p wrela-egraph --release
    /// -- --ignored`; the release `wrela` and `wrela-host` must be built: tools/built).
    #[test]
    #[ignore = "needs a GPU"]
    fn rules_hold_on_the_gpu() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let dir = std::env::temp_dir().join(format!("wrela-egg-rules-{}", std::process::id()));
        let pkg = dir.join("pkg");
        std::fs::create_dir_all(&pkg).unwrap();
        std::fs::write(pkg.join("wrela.toml"), "[package]\nname = \"egg_rules\"\n").unwrap();
        let cs = cases();
        std::fs::write(pkg.join("main.wrela"), source(&cs)).unwrap();
        let bin = root.join("toolchain/target/release");
        let consts = std::env::var("HOME").map(|h| format!("{h}/.wrela/consts")).unwrap();
        let built = std::process::Command::new(bin.join("wrela"))
            .args(["build"])
            .arg(&pkg)
            .arg("-o")
            .arg(dir.join("build"))
            .env("WRELA_CONST_CACHE", consts)
            .env_remove("WRELA_EGG")
            .output()
            .unwrap();
        assert!(built.status.success(), "{}", String::from_utf8_lossy(&built.stderr));
        let ran = std::process::Command::new(root.join("tools/gpu"))
            .arg("--host")
            .arg(bin.join("wrela-host"))
            .arg(dir.join("build"))
            .args(["--frames", "1", "--size", "64x64", "--targets"])
            .arg(dir.join("targets"))
            .output()
            .unwrap();
        assert!(ran.status.success(), "{}", String::from_utf8_lossy(&ran.stderr));
        let rows = read_rows(&dir.join("targets"), cs.len() * SAMPLES * ROW);
        let results = check(&cs, &rows);
        let mut report = String::new();
        for r in &results {
            writeln!(
                report,
                "{:6} {:24} {:?}: sides {:.2} ulps (lhs {:.2}, rhs {:.2} from f64){}",
                if r.passed { "ok" } else { "FAILED" },
                r.name,
                r.group,
                r.sides,
                r.lhs_error,
                r.rhs_error,
                if r.skipped > 0 { format!(", {} skipped", r.skipped) } else { String::new() }
            )
            .unwrap();
        }
        let out = std::env::var("WRELA_EGG_RULES_REPORT")
            .map(std::path::PathBuf::from)
            .unwrap_or(dir.join("report.txt"));
        std::fs::write(&out, &report).unwrap();
        eprintln!("{report}");
        assert!(results.iter().all(|r| r.passed), "a rule failed: {}", out.display());
    }

    /// The kernel's buffer from a `--targets` directory: the buffer of the expected size.
    fn read_rows(targets: &std::path::Path, floats: usize) -> Vec<f32> {
        for e in std::fs::read_dir(targets).unwrap() {
            let p = e.unwrap().path();
            let Ok(bytes) = std::fs::read(&p) else { continue };
            if bytes.len() == floats * 4 {
                return bytes
                    .chunks(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
            }
        }
        panic!("no buffer of {} bytes in {}", floats * 4, targets.display());
    }
}
