//! What each e-class knows: its IR type, its rate (how often it changes: a constant, the same
//! for the whole command, or varying between invocations), an interval its value is in when one
//! is known, whether it may be a NaN, and its value when it's a float constant computed exactly.

use crate::lang::{Lit, Node, Uop};
use egg::{Analysis, DidMerge, EGraph, Id, Language};
use std::collections::HashMap;
use wrela_ir::{BinOp, Builtin, Scalar, TypeDef, TypeId, Types};

/// How often a value changes (language.md §18.13), ordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rate {
    /// Known when the shader is built.
    Const,
    /// The same for every invocation of the command (`same::same_values`).
    Uniform,
    Varying,
}

/// A class's facts.
#[derive(Clone, Debug)]
pub struct Data {
    pub ty: TypeId,
    pub rate: Rate,
    /// Every component of a value that isn't a NaN is in `[lo, hi]` (the ends may be infinite).
    pub iv: Option<(f64, f64)>,
    /// Whether it may be a NaN.
    pub nan: bool,
    /// A scalar `f32` it's known to be, computed as an `f32` (exact operations only).
    pub konst: Option<f32>,
}

/// What a value from outside the stretch, or a let kept as written, is.
#[derive(Clone, Copy, Debug)]
pub struct Given {
    pub ty: TypeId,
    pub rate: Rate,
    /// Its §18.11 work, for a kept let.
    pub work: f64,
}

/// The analysis: the module's types (taken from it for the pass and put back) and what each
/// leaf and kept let is.
#[derive(Default)]
pub struct Facts {
    pub types: Types,
    pub given: HashMap<u32, Given>,
}

pub type Graph = EGraph<Node, Facts>;

/// A scalar or vector type's element and width.
pub fn shape(types: &Types, t: TypeId) -> Option<(Scalar, u8)> {
    match types.get(t) {
        TypeDef::Scalar(s) => Some((*s, 1)),
        TypeDef::Vector(s, n) => Some((*s, *n)),
        _ => None,
    }
}

fn sv(types: &mut Types, s: Scalar, n: u8) -> TypeId {
    if n == 1 { types.scalar(s) } else { types.vector_of(s, n) }
}

/// A node's type, from its operands' (`None`: ill-typed, or a kind of node the pass doesn't
/// type, which only a kept let is).
pub fn type_of(
    types: &mut Types,
    given: &HashMap<u32, Given>,
    n: &Node,
    kids: &[TypeId],
) -> Option<TypeId> {
    let sh = |types: &Types, i: usize| shape(types, kids[i]);
    Some(match n {
        Node::Leaf(v) | Node::Opaque(v, _) => given.get(v)?.ty,
        Node::Lit(l) => types.scalar(l.scalar()),
        // As the IR's verifier types them: an operation's operands are of one type (a scalar
        // with a vector is splatted first), a comparison's scalars.
        Node::Bin(op, _) => {
            let op = op.op();
            if matches!(op, BinOp::Shl | BinOp::Shr) {
                return Some(kids[0]);
            }
            let (s, w) = sh(types, 0)?;
            if kids[0] != kids[1] {
                return None;
            }
            match op {
                _ if op.is_comparison() => (w == 1).then(|| types.bool())?,
                BinOp::And | BinOp::Or => (s == Scalar::Bool).then_some(kids[0])?,
                BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => kids[0],
                _ => (s != Scalar::Bool).then_some(kids[0])?,
            }
        }
        Node::Un(_, _) => {
            sh(types, 0)?;
            kids[0]
        }
        Node::Call(b, _) => {
            let (s, _) = sh(types, 0)?;
            if kids.iter().any(|&k| k != kids[0]) || kids.len() != b.arity() {
                return None;
            }
            match b {
                Builtin::Length | Builtin::Distance | Builtin::Dot => types.scalar(s),
                Builtin::AllEqual => types.bool(),
                Builtin::CountOnes | Builtin::LeadingZeros | Builtin::TrailingZeros => types.u32(),
                _ => kids[0],
            }
        }
        Node::Splat(w, _) => sv(types, sh(types, 0)?.0, *w),
        Node::Swizzle(idx, _) => sv(types, sh(types, 0)?.0, idx.len() as u8),
        Node::Extract(k, _) => types.part(kids[0], *k)?,
        Node::Construct(t, _) => TypeId(*t),
        Node::Select(_) => {
            if kids[1] != kids[2] {
                return None;
            }
            kids[1]
        }
    })
}

/// Outward rounding for an interval's ends: relative `rel`, never across zero (rounding keeps a
/// result's sign).
fn widen((lo, hi): (f64, f64), rel: f64) -> (f64, f64) {
    const TINY: f64 = 1e-37;
    let down = |x: f64| {
        if x > 0.0 {
            (x - x * rel - TINY).max(0.0)
        } else {
            x - x.abs() * rel - TINY * f64::from(u8::from(x < 0.0))
        }
    };
    let up = |x: f64| {
        if x < 0.0 {
            (x + x.abs() * rel + TINY).min(0.0)
        } else {
            x + x * rel + TINY * f64::from(u8::from(x > 0.0))
        }
    };
    (down(lo), up(hi))
}

/// An f32 operation's rounding (with slack for a GPU's), and a transcendental's.
const ARITH: f64 = 1.0 / 1048576.0;
const TRANSC: f64 = 1e-5;

fn hull(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0.min(b.0), a.1.max(b.1))
}

fn finite(iv: Option<(f64, f64)>) -> bool {
    iv.is_some_and(|(lo, hi)| lo.is_finite() && hi.is_finite())
}

fn mul_iv(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let p = [a.0 * b.0, a.0 * b.1, a.1 * b.0, a.1 * b.1];
    if p.iter().any(|x| x.is_nan()) {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    (
        p.iter().copied().fold(f64::INFINITY, f64::min),
        p.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// Interval, NaN and constant facts of a node, from its operands' (`d`), for an `f32` scalar or
/// vector node (`width` its components).
fn values(n: &Node, d: &[&Data], width: u8) -> (Option<(f64, f64)>, bool, Option<f32>) {
    let any_nan = d.iter().any(|x| x.nan);
    let iv = |i: usize| d[i].iv;
    let k = |i: usize| d[i].konst;
    let all_finite = d.iter().all(|x| finite(x.iv));
    match n {
        Node::Lit(l) => match l.as_f32() {
            Some(x) if x.is_nan() => (None, true, None),
            Some(x) => (Some((f64::from(x), f64::from(x))), false, Some(x)),
            None => (None, false, None),
        },
        Node::Leaf(_) | Node::Opaque(..) | Node::Construct(..) if d.is_empty() => {
            (None, true, None)
        }
        Node::Splat(..) | Node::Swizzle(..) | Node::Extract(..) => (iv(0), d[0].nan, None),
        Node::Construct(..) => {
            let iv =
                d.iter().try_fold((f64::INFINITY, f64::NEG_INFINITY), |h, x| Some(hull(h, x.iv?)));
            (iv, any_nan, None)
        }
        Node::Select(_) => {
            let iv = iv(1).zip(iv(2)).map(|(a, b)| hull(a, b));
            (iv, d[1].nan || d[2].nan, None)
        }
        Node::Un(Uop::Neg, _) => (iv(0).map(|(lo, hi)| (-hi, -lo)), d[0].nan, k(0).map(|x| -x)),
        Node::Bin(op, _) => {
            let (a, b) = (iv(0), iv(1));
            let konst = k(0).zip(k(1)).and_then(|(x, y)| match op.op() {
                BinOp::Add => Some(x + y),
                BinOp::Sub => Some(x - y),
                BinOp::Mul => Some(x * y),
                BinOp::Div => Some(x / y),
                _ => None,
            });
            let nan = any_nan || !all_finite;
            let square = op.op() == BinOp::Mul && std::ptr::eq(d[0], d[1]);
            let r = match (op.op(), a, b) {
                (BinOp::Mul, Some((lo, hi)), _) if square => {
                    let (l, h) = (lo.abs().min(hi.abs()), lo.abs().max(hi.abs()));
                    let l = if lo <= 0.0 && hi >= 0.0 { 0.0 } else { l };
                    Some(widen((l * l, h * h), ARITH))
                }
                (BinOp::Add, Some(a), Some(b)) => Some(widen((a.0 + b.0, a.1 + b.1), ARITH)),
                (BinOp::Sub, Some(a), Some(b)) => Some(widen((a.0 - b.1, a.1 - b.0), ARITH)),
                (BinOp::Mul, Some(a), Some(b)) => Some(widen(mul_iv(a, b), ARITH)),
                (BinOp::Div, Some(a), Some(b)) if b.0 > 0.0 || b.1 < 0.0 => {
                    Some(widen(mul_iv(a, (1.0 / b.1, 1.0 / b.0)), ARITH * 4.0))
                }
                // A square: never negative.
                (BinOp::Mul, None, None) if square => Some((0.0, f64::INFINITY)),
                _ => None,
            };
            let nan = match op.op() {
                BinOp::Div => any_nan || !(all_finite && b.is_some_and(|b| b.0 > 0.0 || b.1 < 0.0)),
                BinOp::Add | BinOp::Sub | BinOp::Mul => nan,
                _ => true,
            };
            (r, nan || konst.is_some_and(f32::is_nan), konst.filter(|x| !x.is_nan()))
        }
        Node::Call(b, _) => call_values(*b, d, width, any_nan),
        _ => (None, true, None),
    }
}

fn call_values(
    b: Builtin,
    d: &[&Data],
    width: u8,
    any_nan: bool,
) -> (Option<(f64, f64)>, bool, Option<f32>) {
    let iv = |i: usize| d[i].iv;
    let k = |i: usize| d[i].konst;
    let unit = Some((0.0, 1.0));
    let inf = f64::INFINITY;
    match b {
        Builtin::Abs => {
            let r = match iv(0) {
                Some((lo, hi)) if lo >= 0.0 => (lo, hi),
                Some((lo, hi)) if hi <= 0.0 => (-hi, -lo),
                Some((lo, hi)) => (0.0, (-lo).max(hi)),
                None => (0.0, inf),
            };
            (Some(r), d[0].nan, k(0).map(f32::abs))
        }
        Builtin::Sqrt => match iv(0) {
            Some((lo, hi)) if lo >= 0.0 => {
                (Some(widen((lo.sqrt(), hi.sqrt()), ARITH)), d[0].nan, None)
            }
            _ => (Some((0.0, inf)), true, None),
        },
        Builtin::InverseSqrt => match iv(0) {
            Some((lo, hi)) if lo > 0.0 => {
                (Some(widen((1.0 / hi.sqrt(), 1.0 / lo.sqrt()), TRANSC)), d[0].nan, None)
            }
            _ => (Some((0.0, inf)), true, None),
        },
        Builtin::Min | Builtin::Max => {
            let max = b == Builtin::Max;
            let r = match (iv(0), iv(1)) {
                (Some(a), Some(c)) if max => Some((a.0.max(c.0), a.1.max(c.1))),
                (Some(a), Some(c)) => Some((a.0.min(c.0), a.1.min(c.1))),
                (Some(a), None) | (None, Some(a)) if max => Some((a.0, inf)),
                (Some(a), None) | (None, Some(a)) => Some((-inf, a.1)),
                (None, None) => None,
            };
            let konst = k(0).zip(k(1)).map(|(x, y)| if max { x.max(y) } else { x.min(y) });
            (r, any_nan, konst)
        }
        Builtin::Clamp => {
            let r = iv(1).zip(iv(2)).map(|(lo, hi)| (lo.0, hi.1));
            (r, any_nan, None)
        }
        Builtin::Saturate | Builtin::Fract | Builtin::Step | Builtin::Smoothstep => {
            (unit, any_nan, None)
        }
        Builtin::Sin | Builtin::Cos | Builtin::Sign => (Some((-1.0, 1.0)), any_nan, None),
        Builtin::Normalize => (Some((-1.0, 1.0)), true, None),
        Builtin::Exp | Builtin::Exp2 => {
            let e = |x: f64| if b == Builtin::Exp { x.exp() } else { x.exp2() };
            let r = iv(0).map_or((0.0, inf), |(lo, hi)| widen((e(lo), e(hi)), TRANSC));
            (Some(r), d[0].nan, None)
        }
        Builtin::Log | Builtin::Log2 => {
            let l = |x: f64| if b == Builtin::Log { x.ln() } else { x.log2() };
            match iv(0) {
                Some((lo, hi)) if lo > 0.0 => (Some(widen((l(lo), l(hi)), TRANSC)), d[0].nan, None),
                Some((lo, hi)) if lo >= 0.0 => (Some((-inf, l(hi) + TRANSC)), d[0].nan, None),
                _ => (None, true, None),
            }
        }
        Builtin::Floor | Builtin::Ceil => {
            let r = iv(0).map(|(lo, hi)| (lo.floor(), hi.ceil()));
            (r, d[0].nan, if b == Builtin::Floor { k(0).map(f32::floor) } else { None })
        }
        Builtin::Length | Builtin::Distance => {
            let r = match (b, iv(0)) {
                (Builtin::Length, Some((lo, hi))) => {
                    let m = lo.abs().max(hi.abs());
                    widen((0.0, m * f64::from(width).sqrt()), TRANSC)
                }
                _ => (0.0, inf),
            };
            (Some(r), any_nan, None)
        }
        Builtin::Dot => {
            let r = iv(0).zip(iv(1)).map(|(a, c)| {
                let p = mul_iv(a, c);
                let n = f64::from(width);
                widen((p.0 * n, p.1 * n), ARITH * 4.0)
            });
            // `dot(x, x)` is never negative.
            let r = if std::ptr::eq(d[0], d[1]) {
                Some(r.map_or((0.0, inf), |(_, hi)| (0.0, hi)))
            } else {
                r
            };
            (r, any_nan || !d.iter().all(|x| finite(x.iv)), None)
        }
        Builtin::Mix => {
            let r = match iv(2) {
                Some((lo, hi)) if lo >= 0.0 && hi <= 1.0 => {
                    iv(0).zip(iv(1)).map(|(a, c)| widen(hull(a, c), ARITH * 4.0))
                }
                _ => None,
            };
            (r, any_nan || r.is_none(), None)
        }
        Builtin::Pow => match iv(0) {
            Some((lo, _)) if lo >= 0.0 => (Some((0.0, inf)), any_nan, None),
            _ => (None, true, None),
        },
        _ => (None, true, None),
    }
}

impl Analysis<Node> for Facts {
    type Data = Data;

    fn make(eg: &mut Graph, n: &Node) -> Data {
        let kids: Vec<TypeId> = n.children().iter().map(|&c| eg[c].data.ty).collect();
        let Facts { types, given } = &mut eg.analysis;
        let ty = type_of(types, given, n, &kids).unwrap_or_else(|| {
            panic!("an ill-typed node {n:?} of {kids:?}: a rule's guard let it through")
        });
        let rate = match n {
            Node::Leaf(v) | Node::Opaque(v, _) => given[v].rate,
            Node::Lit(_) => Rate::Const,
            _ => n.children().iter().map(|&c| eg[c].data.rate).max().unwrap_or(Rate::Const),
        };
        let float = shape(&eg.analysis.types, ty).filter(|(s, _)| *s == Scalar::F32);
        let (iv, nan, konst) = match (float, n) {
            (Some(_), Node::Leaf(_) | Node::Opaque(..)) => (None, true, None),
            (Some((_, w)), _) => {
                let ds: Vec<&Data> = n.children().iter().map(|&c| &eg[c].data).collect();
                // A square's two operands are one class: `values` sees them as one `Data`.
                let (iv, nan, konst) = values(n, &ds, w);
                (iv, nan, if w == 1 { konst } else { None })
            }
            (None, _) => (None, true, None),
        };
        Data { ty, rate, iv, nan, konst }
    }

    fn merge(&mut self, a: &mut Data, b: Data) -> DidMerge {
        debug_assert_eq!(a.ty, b.ty, "classes of different types merged");
        let before = (a.rate, a.iv, a.nan, a.konst);
        a.rate = a.rate.min(b.rate);
        a.iv = match (a.iv, b.iv) {
            (Some(x), Some(y)) => {
                let i = (x.0.max(y.0), x.1.min(y.1));
                // An approximate rule may join values a rounding apart: never an empty interval.
                Some(if i.0 <= i.1 { i } else { hull(x, y) })
            }
            (x, y) => x.or(y),
        };
        a.nan &= b.nan;
        a.konst = a.konst.or(b.konst);
        let changed_a = before != (a.rate, a.iv, a.nan, a.konst);
        let changed_b = (b.rate, b.iv, b.nan, b.konst) != (a.rate, a.iv, a.nan, a.konst);
        DidMerge(changed_a, changed_b)
    }

    fn modify(eg: &mut Graph, id: Id) {
        // A float scalar known exactly is a literal (constant folding).
        if let Some(x) = eg[id].data.konst
            && shape(&eg.analysis.types, eg[id].data.ty) == Some((Scalar::F32, 1))
        {
            let lit = eg.add(Node::Lit(Lit::f32(x)));
            eg.union(id, lit);
        }
    }
}
