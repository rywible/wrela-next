//! Where a float value lies, as far as a function's own code says: the least and the most every
//! component of it can be, where its definition bounds it. The writers ask it of a float they
//! convert to an integer: where it lies inside the integer's range, they leave out the clamp the
//! conversion otherwise takes (language.md §11.3).
//!
//! What bounds a value: a constant; `clamp` (and `saturate`) to bounds that are; `min` and `max`
//! where both sides are bounded; `floor`, `ceil`, `trunc`, `round` and `fract`; a splat, a vector
//! built of bounded parts, a component or a swizzle of a bounded vector; and `+`, `-`, `*` of
//! bounded values. Anything else (a load, a parameter, a call) is unbounded.
//!
//! A NaN: a bound here holds as much as the conversion's own clamp does, since that clamp is the
//! same `clamp` a bound came from (`clamp(NaN, lo, hi)` is whatever the target's `clamp` makes
//! of it, inside the conversion or before it). So a value bounded only by arithmetic of
//! constants is bounded for real; one bounded by a clamp is as bounded as the clamp made it.

use crate::*;
use std::collections::HashMap;

/// A function's float values' bounds, found once and asked of many values.
pub struct Ranges<'f> {
    defs: HashMap<ValueId, &'f Expr>,
    memo: HashMap<ValueId, Option<(f64, f64)>>,
}

impl<'f> Ranges<'f> {
    pub fn new(f: &'f Function) -> Ranges<'f> {
        Ranges { defs: visit::definitions(&f.body), memo: HashMap::new() }
    }

    /// The least and the most every component of `v` can be, if its definition bounds it.
    pub fn of(&mut self, v: ValueId) -> Option<(f64, f64)> {
        self.at(v, 0)
    }

    /// Whether `v` lies in `[lo, hi]`, every component.
    pub fn within(&mut self, v: ValueId, lo: f64, hi: f64) -> bool {
        self.of(v).is_some_and(|(a, b)| a >= lo && b <= hi)
    }

    fn at(&mut self, v: ValueId, depth: u32) -> Option<(f64, f64)> {
        if let Some(r) = self.memo.get(&v) {
            return *r;
        }
        if depth > 64 {
            return None;
        }
        let r = self.compute(v, depth);
        self.memo.insert(v, r);
        r
    }

    fn compute(&mut self, v: ValueId, depth: u32) -> Option<(f64, f64)> {
        let e = *self.defs.get(&v)?;
        let mut go = |x: ValueId| self.at(x, depth + 1);
        let r = match e {
            Expr::Const(Const::F32(x)) if x.is_finite() => (f64::from(*x), f64::from(*x)),
            Expr::Splat(x, _) | Expr::Extract(x, _) | Expr::Swizzle(x, _) => go(*x)?,
            Expr::Construct(_, xs) => {
                let mut r = (f64::INFINITY, f64::NEG_INFINITY);
                for &x in xs {
                    let (a, b) = go(x)?;
                    r = (r.0.min(a), r.1.max(b));
                }
                r
            }
            Expr::Builtin(b, xs) => match (b, xs.as_slice()) {
                (Builtin::Clamp, [_, lo, hi]) => {
                    let (lo, hi) = (go(*lo)?, go(*hi)?);
                    if lo.1 > hi.0 {
                        return None;
                    }
                    (lo.0, hi.1)
                }
                (Builtin::Saturate, [_]) => (0.0, 1.0),
                (Builtin::Min, [a, b]) => {
                    let (a, b) = (go(*a)?, go(*b)?);
                    (a.0.min(b.0), a.1.min(b.1))
                }
                (Builtin::Max, [a, b]) => {
                    let (a, b) = (go(*a)?, go(*b)?);
                    (a.0.max(b.0), a.1.max(b.1))
                }
                (Builtin::Floor, [x]) => {
                    let (a, b) = go(*x)?;
                    (a.floor(), b.floor())
                }
                (Builtin::Ceil, [x]) => {
                    let (a, b) = go(*x)?;
                    (a.ceil(), b.ceil())
                }
                (Builtin::Trunc, [x]) => {
                    let (a, b) = go(*x)?;
                    (a.trunc(), b.trunc())
                }
                (Builtin::Round, [x]) => {
                    let (a, b) = go(*x)?;
                    (a.round(), b.round())
                }
                (Builtin::Fract, [_]) => (0.0, 1.0),
                _ => return None,
            },
            Expr::Binary(op, a, b) => {
                let (a, b) = (go(*a)?, go(*b)?);
                // Rounding: a float sum or product may land past the real one's bound; widen by
                // a relative step past f32's, so a bound is never too tight.
                let pad = |x: f64| x.abs() * 1.0e-6 + 1.0e-30;
                let r = match op {
                    BinOp::Add => (a.0 + b.0, a.1 + b.1),
                    BinOp::Sub => (a.0 - b.1, a.1 - b.0),
                    BinOp::Mul => {
                        let ps = [a.0 * b.0, a.0 * b.1, a.1 * b.0, a.1 * b.1];
                        (
                            ps.iter().copied().fold(f64::INFINITY, f64::min),
                            ps.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                        )
                    }
                    _ => return None,
                };
                (r.0 - pad(r.0), r.1 + pad(r.1))
            }
            _ => return None,
        };
        (r.0.is_finite() && r.1.is_finite()).then_some(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_floor_of_a_clamp_is_bounded_and_a_load_is_not() {
        let mut f = Function::new("f", Vec::new(), None);
        let v = ValueId;
        let l = LocalId(0);
        f.body = vec![
            Stmt::Let(v(0), Expr::Load(Place::local(l))),
            Stmt::Let(v(1), Expr::Const(Const::F32(0.0))),
            Stmt::Let(v(2), Expr::Const(Const::F32(7.9999))),
            Stmt::Let(v(3), Expr::Builtin(Builtin::Clamp, vec![v(0), v(1), v(2)])),
            Stmt::Let(v(4), Expr::Builtin(Builtin::Floor, vec![v(3)])),
            Stmt::Let(v(5), Expr::Binary(BinOp::Mul, v(4), v(2))),
        ];
        let mut r = Ranges::new(&f);
        assert!(r.of(v(0)).is_none());
        assert!(r.within(v(4), 0.0, 7.0));
        assert!(!r.within(v(4), 0.0, 6.0));
        assert!(r.within(v(5), -0.001, 56.0));
        assert!(!r.within(v(5), -0.001, 55.0));
    }
}
