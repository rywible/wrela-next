//! Hoisted uniform work (the e-graph spike, wrela-notes/next/backend.md "E-graph spike, round
//! 2"): values a shader would compute the same in every invocation of a command, computed once
//! by the host from the command's uniform bytes and appended to them, so the shader reads them.
//!
//! A [`Tape`] is a list of operations over the program's uniform bytes (as the command carries
//! them, `program_size` long) whose chosen results are written after them, at byte offsets of
//! the shader's extended uniform block (`size` long). The compiler writes a pipeline's tape beside
//! its shader (`pipeline_N.hoist.json`); a host that reads it extends every command's uniform
//! bytes with [`Tape::run`]. Floats are computed as `f32`s by Rust's own operations: GPU results
//! are presentation (language.md §11.3), and these are within a GPU's own error of its.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    F32,
    U32,
    I32,
    Bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fun {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    Not,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Min,
    Max,
    Abs,
    Sqrt,
    InverseSqrt,
    Exp,
    Exp2,
    Log,
    Log2,
    Pow,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Sinh,
    Cosh,
    Tanh,
    Floor,
    Ceil,
    Fract,
    Round,
    Trunc,
    Sign,
    Clamp,
    Saturate,
    Mix,
    Step,
    Smoothstep,
    Length,
    Distance,
    Dot,
    Normalize,
    Cross,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Op {
    /// `comps` components of `kind` at byte `offset` of the program's uniform bytes.
    Load {
        offset: u32,
        kind: Kind,
        comps: u8,
    },
    /// A scalar: its bits (a bool's 0 or 1).
    Const {
        kind: Kind,
        bits: u32,
    },
    Apply {
        f: Fun,
        args: Vec<u32>,
    },
    Splat {
        arg: u32,
        n: u8,
    },
    Swizzle {
        arg: u32,
        idx: Vec<u8>,
    },
    Extract {
        arg: u32,
        k: u8,
    },
    /// A vector of its arguments' components, in order.
    Construct {
        args: Vec<u32>,
    },
    Select {
        cond: u32,
        if_true: u32,
        if_false: u32,
    },
    Convert {
        arg: u32,
        to: Kind,
    },
}

/// Where an operation's value goes: its components as 4-byte words from `offset` (a bool as a
/// `u32` 0 or 1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Output {
    pub op: u32,
    pub offset: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tape {
    pub program_size: u32,
    pub size: u32,
    pub ops: Vec<Op>,
    pub outputs: Vec<Output>,
}

/// A value: its kind and each component's bits.
#[derive(Clone, Debug)]
struct V {
    kind: Kind,
    c: Vec<u32>,
}

impl V {
    fn f(&self, i: usize) -> f32 {
        f32::from_bits(self.c[if self.c.len() == 1 { 0 } else { i }])
    }
    fn u(&self, i: usize) -> u32 {
        self.c[if self.c.len() == 1 { 0 } else { i }]
    }
}

fn floats(n: usize, f: impl Fn(usize) -> f32) -> V {
    V { kind: Kind::F32, c: (0..n).map(|i| f(i).to_bits()).collect() }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn apply(f: Fun, a: &[&V]) -> V {
    let n = a.iter().map(|v| v.c.len()).max().unwrap_or(1);
    let k = a[0].kind;
    let bools = |g: &dyn Fn(usize) -> bool| V {
        kind: Kind::Bool,
        c: (0..n).map(|i| u32::from(g(i))).collect(),
    };
    let ints = |g: &dyn Fn(u32, u32) -> u32| V {
        kind: k,
        c: (0..n).map(|i| g(a[0].u(i), a.get(1).map_or(0, |b| b.u(i)))).collect(),
    };
    let fl = |g: &dyn Fn(f32) -> f32| floats(n, |i| g(a[0].f(i)));
    let fl2 = |g: &dyn Fn(f32, f32) -> f32| floats(n, |i| g(a[0].f(i), a[1].f(i)));
    let cmp = |o: std::cmp::Ordering, eq: bool| -> Box<dyn Fn(usize) -> bool + '_> {
        Box::new(move |i| {
            let ord = match k {
                Kind::F32 => a[0].f(i).partial_cmp(&a[1].f(i)),
                Kind::I32 => Some((a[0].u(i) as i32).cmp(&(a[1].u(i) as i32))),
                _ => Some(a[0].u(i).cmp(&a[1].u(i))),
            };
            ord.is_some_and(|x| x == o || (eq && x == std::cmp::Ordering::Equal))
        })
    };
    use Fun as F;
    use std::cmp::Ordering::{Greater, Less};
    match (f, k) {
        (F::Add, Kind::F32) => fl2(&|x, y| x + y),
        (F::Sub, Kind::F32) => fl2(&|x, y| x - y),
        (F::Mul, Kind::F32) => fl2(&|x, y| x * y),
        (F::Div, Kind::F32) => fl2(&|x, y| x / y),
        (F::Rem, Kind::F32) => fl2(&|x, y| x - y * (x / y).trunc()),
        (F::Neg, Kind::F32) => fl(&|x| -x),
        (F::Add, _) => ints(&|x, y| x.wrapping_add(y)),
        (F::Sub, _) => ints(&|x, y| x.wrapping_sub(y)),
        (F::Mul, _) => ints(&|x, y| x.wrapping_mul(y)),
        (F::Div, Kind::I32) => {
            ints(&|x, y| if y == 0 { x } else { (x as i32).wrapping_div(y as i32) as u32 })
        }
        (F::Div, _) => ints(&|x, y| x.checked_div(y).unwrap_or(x)),
        (F::Rem, Kind::I32) => {
            ints(&|x, y| if y == 0 { 0 } else { (x as i32).wrapping_rem(y as i32) as u32 })
        }
        (F::Rem, _) => ints(&|x, y| if y == 0 { 0 } else { x % y }),
        (F::Neg, _) => ints(&|x, _| x.wrapping_neg()),
        (F::Not, Kind::Bool) => ints(&|x, _| u32::from(x == 0)),
        (F::Not, _) => ints(&|x, _| !x),
        (F::Lt, _) => bools(&*cmp(Less, false)),
        (F::Le, _) => bools(&*cmp(Less, true)),
        (F::Gt, _) => bools(&*cmp(Greater, false)),
        (F::Ge, _) => bools(&*cmp(Greater, true)),
        (F::Eq, Kind::F32) => bools(&|i| a[0].f(i) == a[1].f(i)),
        (F::Ne, Kind::F32) => bools(&|i| a[0].f(i) != a[1].f(i)),
        (F::Eq, _) => bools(&|i| a[0].u(i) == a[1].u(i)),
        (F::Ne, _) => bools(&|i| a[0].u(i) != a[1].u(i)),
        (F::And | F::BitAnd, _) => ints(&|x, y| x & y),
        (F::Or | F::BitOr, _) => ints(&|x, y| x | y),
        (F::BitXor, _) => ints(&|x, y| x ^ y),
        (F::Shl, _) => ints(&|x, y| x.wrapping_shl(y)),
        (F::Shr, Kind::I32) => ints(&|x, y| ((x as i32).wrapping_shr(y)) as u32),
        (F::Shr, _) => ints(&|x, y| x.wrapping_shr(y)),
        (F::Min, Kind::F32) => fl2(&f32::min),
        (F::Max, Kind::F32) => fl2(&f32::max),
        (F::Min, Kind::I32) => ints(&|x, y| (x as i32).min(y as i32) as u32),
        (F::Max, Kind::I32) => ints(&|x, y| (x as i32).max(y as i32) as u32),
        (F::Min, _) => ints(&|x, y| x.min(y)),
        (F::Max, _) => ints(&|x, y| x.max(y)),
        (F::Abs, Kind::F32) => fl(&f32::abs),
        (F::Abs, _) => ints(&|x, _| (x as i32).unsigned_abs()),
        (F::Sqrt, _) => fl(&f32::sqrt),
        (F::InverseSqrt, _) => fl(&|x| 1.0 / x.sqrt()),
        (F::Exp, _) => fl(&f32::exp),
        (F::Exp2, _) => fl(&f32::exp2),
        (F::Log, _) => fl(&f32::ln),
        (F::Log2, _) => fl(&f32::log2),
        (F::Pow, _) => fl2(&f32::powf),
        (F::Sin, _) => fl(&f32::sin),
        (F::Cos, _) => fl(&f32::cos),
        (F::Tan, _) => fl(&f32::tan),
        (F::Asin, _) => fl(&f32::asin),
        (F::Acos, _) => fl(&f32::acos),
        (F::Atan, _) => fl(&f32::atan),
        (F::Atan2, _) => fl2(&f32::atan2),
        (F::Sinh, _) => fl(&f32::sinh),
        (F::Cosh, _) => fl(&f32::cosh),
        (F::Tanh, _) => fl(&f32::tanh),
        (F::Floor, _) => fl(&f32::floor),
        (F::Ceil, _) => fl(&f32::ceil),
        (F::Fract, _) => fl(&|x| x - x.floor()),
        (F::Round, _) => fl(&f32::round_ties_even),
        (F::Trunc, _) => fl(&f32::trunc),
        (F::Sign, _) => fl(&|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }),
        (F::Clamp, _) => floats(n, |i| a[0].f(i).max(a[1].f(i)).min(a[2].f(i))),
        (F::Saturate, _) => fl(&|x| x.clamp(0.0, 1.0)),
        (F::Mix, _) => floats(n, |i| a[0].f(i) + (a[1].f(i) - a[0].f(i)) * a[2].f(i)),
        (F::Step, _) => floats(n, |i| if a[1].f(i) >= a[0].f(i) { 1.0 } else { 0.0 }),
        (F::Smoothstep, _) => floats(n, |i| smoothstep(a[0].f(i), a[1].f(i), a[2].f(i))),
        (F::Dot, _) => floats(1, |_| (0..a[0].c.len()).map(|i| a[0].f(i) * a[1].f(i)).sum()),
        (F::Length, _) => {
            floats(1, |_| (0..a[0].c.len()).map(|i| a[0].f(i) * a[0].f(i)).sum::<f32>().sqrt())
        }
        (F::Distance, _) => floats(1, |_| {
            (0..a[0].c.len()).map(|i| (a[0].f(i) - a[1].f(i)).powi(2)).sum::<f32>().sqrt()
        }),
        (F::Normalize, _) => {
            let l = (0..a[0].c.len()).map(|i| a[0].f(i) * a[0].f(i)).sum::<f32>().sqrt();
            floats(a[0].c.len(), |i| a[0].f(i) / l)
        }
        (F::Cross, _) => floats(3, |i| {
            let (j, k) = ((i + 1) % 3, (i + 2) % 3);
            a[0].f(j) * a[1].f(k) - a[0].f(k) * a[1].f(j)
        }),
    }
}

fn convert(v: &V, to: Kind) -> V {
    let c =
        v.c.iter()
            .map(|&b| match (v.kind, to) {
                (x, y) if x == y => b,
                (Kind::F32, Kind::U32) => {
                    let f = f32::from_bits(b);
                    if f.is_nan() { 0 } else { f as u32 }
                }
                (Kind::F32, Kind::I32) => {
                    let f = f32::from_bits(b);
                    if f.is_nan() { 0 } else { f as i32 as u32 }
                }
                (Kind::U32, Kind::F32) => (b as f32).to_bits(),
                (Kind::I32, Kind::F32) => (b as i32 as f32).to_bits(),
                (Kind::Bool, Kind::F32) => if b != 0 { 1.0f32 } else { 0.0 }.to_bits(),
                (_, Kind::Bool) => u32::from(b != 0),
                (Kind::Bool, _) => u32::from(b != 0),
                (Kind::F32, Kind::F32) => b,
                _ => b,
            })
            .collect();
    V { kind: to, c }
}

impl Tape {
    /// The command's uniform bytes, extended with the tape's values (`size` bytes).
    pub fn run(&self, bytes: &[u8]) -> Vec<u8> {
        let mut out = bytes.to_vec();
        out.resize(self.size as usize, 0);
        let word =
            |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        let mut vals: Vec<V> = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let v = match op {
                Op::Load { offset, kind, comps } => V {
                    kind: *kind,
                    c: (0..usize::from(*comps)).map(|i| word(*offset as usize + 4 * i)).collect(),
                },
                Op::Const { kind, bits } => V { kind: *kind, c: vec![*bits] },
                Op::Apply { f, args } => {
                    let a: Vec<&V> = args.iter().map(|&i| &vals[i as usize]).collect();
                    apply(*f, &a)
                }
                Op::Splat { arg, n } => {
                    let a = &vals[*arg as usize];
                    V { kind: a.kind, c: vec![a.c[0]; usize::from(*n)] }
                }
                Op::Swizzle { arg, idx } => {
                    let a = &vals[*arg as usize];
                    V { kind: a.kind, c: idx.iter().map(|&i| a.c[usize::from(i)]).collect() }
                }
                Op::Extract { arg, k } => {
                    let a = &vals[*arg as usize];
                    V { kind: a.kind, c: vec![a.c[usize::from(*k)]] }
                }
                Op::Construct { args } => {
                    let kind = vals[args[0] as usize].kind;
                    V { kind, c: args.iter().flat_map(|&i| vals[i as usize].c.clone()).collect() }
                }
                Op::Select { cond, if_true, if_false } => {
                    let (c, t, f) = (
                        &vals[*cond as usize],
                        &vals[*if_true as usize],
                        &vals[*if_false as usize],
                    );
                    let n = t.c.len().max(f.c.len());
                    V {
                        kind: t.kind,
                        c: (0..n).map(|i| if c.u(i) != 0 { t.u(i) } else { f.u(i) }).collect(),
                    }
                }
                Op::Convert { arg, to } => convert(&vals[*arg as usize], *to),
            };
            vals.push(v);
        }
        for o in &self.outputs {
            let v = &vals[o.op as usize];
            for (i, w) in v.c.iter().enumerate() {
                let at = o.offset as usize + 4 * i;
                out[at..at + 4].copy_from_slice(&w.to_le_bytes());
            }
        }
        out
    }
}

fn registry() -> &'static Mutex<HashMap<u64, Arc<Tape>>> {
    static R: OnceLock<Mutex<HashMap<u64, Arc<Tape>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn key(shader: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    shader.hash(&mut h);
    h.finish()
}

/// Records the tape written beside a shader (by its bytes), for the host that builds it.
pub fn register(shader: &[u8], tape: Tape) {
    registry().lock().expect("the tapes").insert(key(shader), Arc::new(tape));
}

/// The tape for a shader, if its build has one.
pub fn lookup(shader: &[u8]) -> Option<Arc<Tape>> {
    registry().lock().expect("the tapes").get(&key(shader)).cloned()
}

/// The file beside a shader that holds its tape.
pub fn file_for(shader: &str) -> String {
    format!("{shader}.hoist.json")
}

/// A tape from its file's text.
pub fn parse(text: &str) -> Result<Tape, String> {
    serde_json::from_str(text).map_err(|e| e.to_string())
}

/// A tape as its file's text.
pub fn write(t: &Tape) -> String {
    serde_json::to_string(t).expect("a tape serializes")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `smoothstep(u.a, u.b, 0.5) * u.c` from the uniform bytes, written after them.
    #[test]
    fn a_tape_extends_the_uniforms() {
        let words = [0.0f32, 1.0, 4.0, 0.0];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let tape = Tape {
            program_size: 16,
            size: 32,
            ops: vec![
                Op::Load { offset: 0, kind: Kind::F32, comps: 1 },
                Op::Load { offset: 4, kind: Kind::F32, comps: 1 },
                Op::Const { kind: Kind::F32, bits: 0.5f32.to_bits() },
                Op::Apply { f: Fun::Smoothstep, args: vec![0, 1, 2] },
                Op::Load { offset: 8, kind: Kind::F32, comps: 1 },
                Op::Apply { f: Fun::Mul, args: vec![3, 4] },
                Op::Apply { f: Fun::Lt, args: vec![0, 1] },
            ],
            outputs: vec![Output { op: 5, offset: 16 }, Output { op: 6, offset: 20 }],
        };
        let out = tape.run(&bytes);
        assert_eq!(out.len(), 32);
        assert_eq!(&out[..16], &bytes[..]);
        assert_eq!(f32::from_le_bytes(out[16..20].try_into().unwrap()), 2.0);
        assert_eq!(u32::from_le_bytes(out[20..24].try_into().unwrap()), 1);
        assert_eq!(parse(&write(&tape)).unwrap(), tape);
    }
}
