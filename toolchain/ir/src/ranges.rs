//! Static float and integer range analysis (with flow-sensitive branch narrowing, early-exit
//! guards, relational buffer-length facts, and canonical `for`-loop induction bounds).
//!
//! Consumers:
//! - Shader backends (`metal.rs`, `spirv.rs`): omit float-to-int clamping when [`Ranges::within`]
//!   proves the float already lies inside the target integer type's range (language.md §11.3).
//! - GPU debug bounds checks (`bounds.rs`): erase `if i >= len` checks whenever [`FlowEnv`] proves
//!   `0 <= i < len` for fixed arrays (`Len::Const`) or storage buffers (`Len::Of`).
//! - IR optimizer (`opt.rs`): fold statically settled comparisons, erase dead branches, and
//!   eliminate redundant `abs`/`clamp`/`saturate`/`min`/`max` guards on proven-finite ranges.
//!
//! A NaN: a float range here is a fact about finite values. An operation whose result is bounded
//! whatever its argument (`sin`, `cos`, `atan`, `sign`, `fract`, `smoothstep`) is bounded only
//! when its arguments are, since a NaN or an infinity in gives a NaN out (`sin(inf)`,
//! `fract(inf)`), and erasing a guard on it would let the NaN through. A clamp's bound holds as
//! much as the clamp does (`clamp(NaN, lo, hi)` is whatever the target's `clamp` makes of it, and
//! a guard erased after it is the same clamp again); `step` compares, so it is 0 or 1 whatever
//! comes in.

use crate::*;
use std::collections::{HashMap, HashSet};

/// Closed integer interval `[lo, hi]` in arbitrary `i128` precision (covers all `i8..u64`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntRange {
    pub lo: i128,
    pub hi: i128,
}

impl IntRange {
    pub fn point(v: i128) -> Self {
        Self { lo: v, hi: v }
    }

    pub fn of_scalar(s: Scalar) -> Option<Self> {
        Some(match s {
            Scalar::I8 => Self { lo: i128::from(i8::MIN), hi: i128::from(i8::MAX) },
            Scalar::U8 => Self { lo: 0, hi: i128::from(u8::MAX) },
            Scalar::I16 => Self { lo: i128::from(i16::MIN), hi: i128::from(i16::MAX) },
            Scalar::U16 => Self { lo: 0, hi: i128::from(u16::MAX) },
            Scalar::I32 => Self { lo: i128::from(i32::MIN), hi: i128::from(i32::MAX) },
            Scalar::U32 => Self { lo: 0, hi: i128::from(u32::MAX) },
            Scalar::I64 => Self { lo: i128::from(i64::MIN), hi: i128::from(i64::MAX) },
            Scalar::U64 => Self { lo: 0, hi: i128::from(u64::MAX) },
            _ => return None,
        })
    }

    pub fn intersect(self, other: Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        (lo <= hi).then_some(Self { lo, hi })
    }

    pub fn hull(self, other: Self) -> Self {
        Self { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }
}

/// A function's float values' bounds, found once and asked of many values (expression-level,
/// backward-memoized; used by shader writers for float-to-int conversion clamp elision).
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
        let r = eval_float_expr(e, &mut go)?;
        (r.0.is_finite() && r.1.is_finite() && r.0 <= r.1).then_some(r)
    }
}

/// Evaluates sound float interval bounds for expression `e` given an operand lookup `go`.
fn eval_float_expr(
    e: &Expr,
    go: &mut impl FnMut(ValueId) -> Option<(f64, f64)>,
) -> Option<(f64, f64)> {
    let r = match e {
        Expr::Const(Const::F32(x)) if x.is_finite() => (f64::from(*x), f64::from(*x)),
        Expr::Const(Const::F64(x)) if x.is_finite() => (*x, *x),
        Expr::Const(Const::F16(x)) if x.is_finite() => (f64::from(*x), f64::from(*x)),
        Expr::Zero(_) => (0.0, 0.0),
        Expr::Splat(x, _) | Expr::Extract(x, _) | Expr::Swizzle(x, _) => go(*x)?,
        Expr::Unary(UnOp::Neg, x) => {
            let (a, b) = go(*x)?;
            (-b, -a)
        }
        Expr::Select { if_true, if_false, .. } => {
            let (a0, a1) = go(*if_true)?;
            let (b0, b1) = go(*if_false)?;
            (a0.min(b0), a1.max(b1))
        }
        Expr::Construct(_, xs) if !xs.is_empty() => {
            let mut r = (f64::INFINITY, f64::NEG_INFINITY);
            for &x in xs {
                let (a, b) = go(x)?;
                r = (r.0.min(a), r.1.max(b));
            }
            r
        }
        Expr::Builtin(b, xs) => match (b, xs.as_slice()) {
            (Builtin::Clamp, [x, lo, hi]) => {
                let (lo, hi) = (go(*lo)?, go(*hi)?);
                if lo.1 > hi.0 {
                    return None;
                }
                if let Some((a, b)) = go(*x) {
                    (a.max(lo.0).min(hi.1), b.min(hi.1).max(lo.0))
                } else {
                    (lo.0, hi.1)
                }
            }
            (Builtin::Saturate, [x]) => {
                if let Some((a, b)) = go(*x) {
                    (a.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
                } else {
                    (0.0, 1.0)
                }
            }
            (Builtin::Min, [a, b]) => {
                let (a, b) = (go(*a)?, go(*b)?);
                (a.0.min(b.0), a.1.min(b.1))
            }
            (Builtin::Max, [a, b]) => {
                let (a, b) = (go(*a)?, go(*b)?);
                (a.0.max(b.0), a.1.max(b.1))
            }
            (Builtin::Abs, [x]) => {
                let (a, b) = go(*x)?;
                if a >= 0.0 {
                    (a, b)
                } else if b <= 0.0 {
                    (-b, -a)
                } else {
                    (0.0, (-a).max(b))
                }
            }
            (Builtin::Sqrt, [x]) => {
                let (a, b) = go(*x)?;
                if a < 0.0 {
                    return None;
                }
                let lo = a.sqrt();
                let hi = b.sqrt();
                ((lo * (1.0 - 1.0e-6)).max(0.0), hi * (1.0 + 1.0e-6) + 1.0e-30)
            }
            (Builtin::Sin | Builtin::Cos, [x]) => {
                go(*x)?;
                (-1.0, 1.0)
            }
            (Builtin::Atan, [x]) => {
                go(*x)?;
                (-std::f64::consts::FRAC_PI_2 - 1.0e-5, std::f64::consts::FRAC_PI_2 + 1.0e-5)
            }
            (Builtin::Smoothstep, [e0, e1, x]) => {
                go(*e0)?;
                go(*e1)?;
                go(*x)?;
                (0.0, 1.0)
            }
            (Builtin::Step, [_, _]) => (0.0, 1.0),
            (Builtin::Sign, [x]) => {
                go(*x)?;
                (-1.0, 1.0)
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
            (Builtin::Fract, [x]) => {
                go(*x)?;
                (0.0, 1.0)
            }
            (Builtin::Mix, [a, b, t]) => {
                let (a, b, t) = (go(*a)?, go(*b)?, go(*t)?);
                if t.0 >= 0.0 && t.1 <= 1.0 {
                    let lo = a.0.min(b.0);
                    let hi = a.1.max(b.1);
                    let pad = lo.abs().max(hi.abs()) * 1.0e-5 + 1.0e-30;
                    (lo - pad, hi + pad)
                } else {
                    return None;
                }
            }
            (Builtin::Length, [x]) => {
                let (a, b) = go(*x)?;
                let m = a.abs().max(b.abs());
                // At most 4 components: `||v||₂ <= 2 * max|v_i|`.
                (0.0, 2.0 * m * (1.0 + 1.0e-5) + 1.0e-30)
            }
            _ => return None,
        },
        Expr::Binary(op, a, b) => {
            let (a, b) = (go(*a)?, go(*b)?);
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
                BinOp::Div if b.0 > 0.0 || b.1 < 0.0 => {
                    let ps = [a.0 / b.0, a.0 / b.1, a.1 / b.0, a.1 / b.1];
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
    (r.0.is_finite() && r.1.is_finite() && r.0 <= r.1).then_some(r)
}

/// Flow-sensitive static range environment for a single IR function, tracking:
/// - integer ranges (`IntRange`) and float ranges `(f64, f64)` per SSA value,
/// - relational `0 <= i < ArrayLength(root)` facts for storage buffer bounds-check erasure,
/// - loop induction variable bounds for canonical `for` loops,
/// - branch narrowing (`if`, `&&`, `||`, `!`) and early-exit post-dominator narrowing.
#[derive(Clone, Debug, Default)]
pub struct FlowEnv {
    pub ints: HashMap<ValueId, IntRange>,
    pub floats: HashMap<ValueId, (f64, f64)>,
    /// Values proven strictly non-negative (e.g. `abs`, `sqrt`, `length`, `saturate`) even when
    /// their upper bound is unknown.
    pub nonneg_floats: HashSet<ValueId>,
    /// Values defined by `Expr::ArrayLength(Place { root, path: [] })`.
    pub array_len_of: HashMap<ValueId, PlaceRoot>,
    /// Relational facts: `(v, root)` means `v` is proven `0 <= v < ArrayLength(root)` here.
    pub lt_len: HashSet<(ValueId, PlaceRoot)>,
    /// Upper-bound aliases: `v` is proven `<= ArrayLength(root)` (e.g. `min(count, len(buf))`).
    pub le_len: HashSet<(ValueId, PlaceRoot)>,
    /// Local integer bounds valid at the current program point (cleared on unknown stores or loops).
    pub local_ints: HashMap<LocalId, IntRange>,
    /// Definitions of comparisons and boolean ops for branch narrowing.
    cmp_defs: HashMap<ValueId, (BinOp, ValueId, ValueId)>,
    bool_defs: HashMap<ValueId, (BinOp, ValueId, ValueId)>,
    not_defs: HashMap<ValueId, ValueId>,
    /// Locals whose value was loaded into SSA value `v` (to narrow locals across `if` / loops).
    load_of: HashMap<ValueId, LocalId>,
}

impl FlowEnv {
    pub fn new() -> Self {
        Self::default()
    }

    /// Integer range of `v` (falling back to the full range of its scalar integer type).
    pub fn int_of(&self, types: &Types, f: &Function, v: ValueId) -> Option<IntRange> {
        if let Some(&r) = self.ints.get(&v) {
            return Some(r);
        }
        let s = types.as_scalar(f.value_ty(v))?;
        IntRange::of_scalar(s)
    }

    /// Float range of `v`, if finite and bounded at this point.
    pub fn float_of(&self, v: ValueId) -> Option<(f64, f64)> {
        self.floats.get(&v).copied()
    }

    /// Whether `v` is proven `>= 0.0` (either by finite range or non-negativity fact).
    pub fn is_nonneg_float(&self, v: ValueId) -> bool {
        self.nonneg_floats.contains(&v) || self.floats.get(&v).is_some_and(|&(lo, _)| lo >= 0.0)
    }

    /// Whether index `i` is statically proven within `[0, len)` for a fixed-size array of `len`.
    pub fn in_const_bounds(&self, types: &Types, f: &Function, i: ValueId, len: u32) -> bool {
        if len == 0 {
            return false;
        }
        self.int_of(types, f, i).is_some_and(|r| r.lo >= 0 && r.hi < i128::from(len))
    }

    /// Whether index `i` is statically proven within `[0, ArrayLength(root))`.
    pub fn in_storage_bounds(
        &self,
        types: &Types,
        f: &Function,
        i: ValueId,
        root: &PlaceRoot,
    ) -> bool {
        if !self.lt_len.contains(&(i, root.clone())) {
            return false;
        }
        self.int_of(types, f, i).is_some_and(|r| r.lo >= 0)
    }

    /// Updates `self` for a single `Stmt::Let(v, e)` statement.
    pub fn record_let(&mut self, types: &Types, f: &Function, v: ValueId, e: &Expr) {
        let ty = f.value_ty(v);
        match e {
            Expr::ArrayLength(p) if p.path.is_empty() => {
                self.array_len_of.insert(v, p.root.clone());
                self.le_len.insert((v, p.root.clone()));
            }
            Expr::Load(p) if p.path.is_empty() => {
                if let PlaceRoot::Local(l) = p.root {
                    self.load_of.insert(v, l);
                    if let Some(&r) = self.local_ints.get(&l) {
                        self.ints.insert(v, r);
                    }
                }
            }
            Expr::Binary(op, a, b) if op.is_comparison() => {
                self.cmp_defs.insert(v, (*op, *a, *b));
            }
            Expr::Binary(op @ (BinOp::And | BinOp::Or), a, b) => {
                self.bool_defs.insert(v, (*op, *a, *b));
            }
            Expr::Unary(UnOp::Not, x) => {
                self.not_defs.insert(v, *x);
            }
            _ => {}
        }

        // Propagate relational `<= ArrayLength(root)` through `min(a, b)` on unsigned ints.
        if let Expr::Builtin(Builtin::Min, xs) = e
            && xs.len() == 2
            && matches!(types.as_scalar(ty), Some(Scalar::U32 | Scalar::U64))
        {
            let mut roots = Vec::new();
            for &x in xs {
                for &(val, ref root) in &self.le_len {
                    if val == x {
                        roots.push(root.clone());
                    }
                }
                if let Some(root) = self.array_len_of.get(&x) {
                    roots.push(root.clone());
                }
            }
            for root in roots {
                self.le_len.insert((v, root));
            }
            // Likewise propagate `< ArrayLength(root)` through `min(a, b)` when both >= 0.
            let mut lt_roots = Vec::new();
            for &x in xs {
                for &(val, ref root) in &self.lt_len {
                    if val == x {
                        lt_roots.push(root.clone());
                    }
                }
            }
            for root in lt_roots {
                self.lt_len.insert((v, root));
            }
        }

        // Track integer range if `ty` is an integer scalar.
        if let Some(s) = types.as_scalar(ty)
            && s.is_int()
            && let Some(r) = self.eval_int(types, f, s, e)
        {
            let clipped = IntRange::of_scalar(s).and_then(|lim| r.intersect(lim)).unwrap_or(r);
            self.ints.insert(v, clipped);
        }

        // Track float range & non-negativity.
        if matches!(
            types.get(ty),
            TypeDef::Scalar(Scalar::F32 | Scalar::F64 | Scalar::F16)
                | TypeDef::Vector(Scalar::F32 | Scalar::F64 | Scalar::F16, _)
        ) {
            if let Some(r) = eval_float_expr(e, &mut |x| self.floats.get(&x).copied()) {
                self.floats.insert(v, r);
                if r.0 >= 0.0 {
                    self.nonneg_floats.insert(v);
                }
            } else if matches!(
                e,
                Expr::Builtin(
                    Builtin::Abs
                        | Builtin::Sqrt
                        | Builtin::Length
                        | Builtin::Distance
                        | Builtin::Saturate
                        | Builtin::Smoothstep
                        | Builtin::Exp
                        | Builtin::Exp2
                        | Builtin::Fract,
                    _
                )
            ) {
                self.nonneg_floats.insert(v);
            } else if let Expr::Builtin(Builtin::Max, xs) = e
                && xs.len() == 2
                && (self.is_nonneg_float(xs[0]) || self.is_nonneg_float(xs[1]))
            {
                self.nonneg_floats.insert(v);
            } else if let Expr::Builtin(Builtin::Clamp, xs) = e
                && xs.len() == 3
                && self.is_nonneg_float(xs[1])
            {
                self.nonneg_floats.insert(v);
            }
        }
    }

    fn eval_int(&self, types: &Types, f: &Function, s: Scalar, e: &Expr) -> Option<IntRange> {
        let lim = IntRange::of_scalar(s)?;
        let signed = matches!(s, Scalar::I8 | Scalar::I16 | Scalar::I32 | Scalar::I64);
        let get = |x: ValueId| self.int_of(types, f, x);
        match e {
            Expr::Const(c) => int_const_val(c).map(IntRange::point),
            Expr::Zero(_) => Some(IntRange::point(0)),
            Expr::Load(p) if p.path.is_empty() => match p.root {
                PlaceRoot::Local(l) => self.local_ints.get(&l).copied(),
                _ => None,
            },
            Expr::Select { if_true, if_false, .. } => {
                let (a, b) = (get(*if_true)?, get(*if_false)?);
                Some(a.hull(b))
            }
            Expr::Convert(x, _) => {
                let xt = f.value_ty(*x);
                if let Some(xs) = types.as_scalar(xt) {
                    if xs.is_int() {
                        let r = get(*x)?;
                        return if r.lo >= lim.lo && r.hi <= lim.hi { Some(r) } else { None };
                    }
                    if xs.is_float() {
                        let (fl, fh) = self.floats.get(x).copied()?;
                        let lo = (fl.trunc() as i128).clamp(lim.lo, lim.hi);
                        let hi = (fh.trunc() as i128).clamp(lim.lo, lim.hi);
                        return Some(IntRange { lo, hi });
                    }
                }
                None
            }
            Expr::Bitcast(x, _) => {
                let r = get(*x)?;
                if r.lo >= lim.lo && r.hi <= lim.hi { Some(r) } else { None }
            }
            Expr::Builtin(Builtin::Min, xs) if xs.len() == 2 => {
                let (a, b) = (get(xs[0])?, get(xs[1])?);
                Some(IntRange { lo: a.lo.min(b.lo), hi: a.hi.min(b.hi) })
            }
            Expr::Builtin(Builtin::Max, xs) if xs.len() == 2 => {
                let (a, b) = (get(xs[0])?, get(xs[1])?);
                Some(IntRange { lo: a.lo.max(b.lo), hi: a.hi.max(b.hi) })
            }
            Expr::Builtin(Builtin::Clamp, xs) if xs.len() == 3 => {
                let (lo, hi) = (get(xs[1])?, get(xs[2])?);
                if lo.hi > hi.lo {
                    return None;
                }
                let x = get(xs[0]).unwrap_or(lim);
                Some(IntRange { lo: x.lo.max(lo.lo).min(hi.hi), hi: x.hi.min(hi.hi).max(lo.lo) })
            }
            Expr::Binary(op, a, b) => {
                let (ra, rb) = (get(*a)?, get(*b)?);
                match op {
                    BinOp::BitAnd => {
                        if ra.lo >= 0 && rb.lo >= 0 {
                            Some(IntRange { lo: 0, hi: ra.hi.min(rb.hi) })
                        } else if rb.lo >= 0 {
                            Some(IntRange { lo: 0, hi: rb.hi })
                        } else if ra.lo >= 0 {
                            Some(IntRange { lo: 0, hi: ra.hi })
                        } else {
                            None
                        }
                    }
                    BinOp::Shr if !signed && rb.lo >= 0 && rb.hi < 64 => {
                        Some(IntRange { lo: ra.lo >> rb.hi, hi: ra.hi >> rb.lo })
                    }
                    BinOp::Shl if !signed && rb.lo >= 0 && rb.hi < 32 => {
                        let lo = ra.lo.checked_shl(rb.lo as u32)?;
                        let hi = ra.hi.checked_shl(rb.hi as u32)?;
                        (lo >= lim.lo && hi <= lim.hi).then_some(IntRange { lo, hi })
                    }
                    BinOp::Rem if rb.lo > 0 => {
                        if ra.lo >= 0 {
                            Some(IntRange { lo: 0, hi: ra.hi.min(rb.hi - 1) })
                        } else {
                            Some(IntRange { lo: -(rb.hi - 1), hi: rb.hi - 1 })
                        }
                    }
                    BinOp::Div if rb.lo > 0 && ra.lo >= 0 => {
                        Some(IntRange { lo: ra.lo / rb.hi, hi: ra.hi / rb.lo })
                    }
                    BinOp::Add => {
                        let lo = ra.lo.checked_add(rb.lo)?;
                        let hi = ra.hi.checked_add(rb.hi)?;
                        (lo >= lim.lo && hi <= lim.hi).then_some(IntRange { lo, hi })
                    }
                    BinOp::Sub => {
                        let lo = ra.lo.checked_sub(rb.hi)?;
                        let hi = ra.hi.checked_sub(rb.lo)?;
                        (lo >= lim.lo && hi <= lim.hi).then_some(IntRange { lo, hi })
                    }
                    BinOp::Mul => {
                        let ps = [
                            ra.lo.checked_mul(rb.lo)?,
                            ra.lo.checked_mul(rb.hi)?,
                            ra.hi.checked_mul(rb.lo)?,
                            ra.hi.checked_mul(rb.hi)?,
                        ];
                        let lo = *ps.iter().min()?;
                        let hi = *ps.iter().max()?;
                        (lo >= lim.lo && hi <= lim.hi).then_some(IntRange { lo, hi })
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Updates local state for `Stmt::Store(p, v)`.
    pub fn record_store(&mut self, types: &Types, f: &Function, p: &Place, v: ValueId) {
        if let PlaceRoot::Local(l) = p.root {
            if p.path.is_empty() {
                if let Some(r) = self.int_of(types, f, v) {
                    self.local_ints.insert(l, r);
                } else {
                    self.local_ints.remove(&l);
                }
            } else {
                self.local_ints.remove(&l);
            }
        }
    }

    /// Invalidates locals that may be written by a block (or call with `mut` args).
    pub fn havoc_written(&mut self, m: &Module, b: &[Stmt]) {
        visit::walk(b, &mut |s| match s {
            Stmt::Store(Place { root: PlaceRoot::Local(l), .. }, _) => {
                self.local_ints.remove(l);
            }
            Stmt::Let(_, Expr::Call(g, args)) | Stmt::Eval(Expr::Call(g, args)) => {
                if let Some(callee) = m.functions.get(g.index()) {
                    for (a, p) in args.iter().zip(&callee.params) {
                        if let (Arg::Place(Place { root: PlaceRoot::Local(l), .. }), true) =
                            (a, p.mutable)
                        {
                            self.local_ints.remove(l);
                        }
                    }
                }
            }
            _ => {}
        });
    }

    /// Narrows `self` under the assumption that boolean `cond` evaluates to `truth`.
    pub fn narrow_cond(&mut self, types: &Types, f: &Function, cond: ValueId, truth: bool) {
        if let Some(&inner) = self.not_defs.get(&cond) {
            self.narrow_cond(types, f, inner, !truth);
            return;
        }
        if let Some(&(op, a, b)) = self.bool_defs.get(&cond) {
            if (op == BinOp::And && truth) || (op == BinOp::Or && !truth) {
                self.narrow_cond(types, f, a, truth);
                self.narrow_cond(types, f, b, truth);
            }
            return;
        }
        let Some(&(op, a, b)) = self.cmp_defs.get(&cond) else {
            return;
        };
        let eff = if truth { op } else { negate_cmp(op) };
        let Some(s) = types.as_scalar(f.value_ty(a)) else {
            return;
        };
        if s.is_int() {
            self.narrow_int_cmp(types, f, eff, a, b);
        } else if s.is_float() {
            self.narrow_float_cmp(eff, a, b);
        }
    }

    fn narrow_int_cmp(&mut self, types: &Types, f: &Function, op: BinOp, a: ValueId, b: ValueId) {
        // Relational buffer-length facts: `a < b` where `b <= ArrayLength(root)` => `a < ArrayLength(root)`!
        let mark_lt_len = |env: &mut Self, idx: ValueId, bound: ValueId| {
            let mut roots: Vec<PlaceRoot> =
                env.le_len.iter().filter_map(|(v, r)| (*v == bound).then(|| r.clone())).collect();
            if let Some(r) = env.array_len_of.get(&bound) {
                roots.push(r.clone());
            }
            for r in roots {
                env.lt_len.insert((idx, r));
            }
        };
        match op {
            BinOp::Lt => mark_lt_len(self, a, b),
            BinOp::Gt => mark_lt_len(self, b, a),
            _ => {}
        }

        // Numeric range narrowing for `a` and `b`.
        let ra = self.int_of(types, f, a);
        let rb = self.int_of(types, f, b);
        let (Some(ra), Some(rb)) = (ra, rb) else {
            return;
        };
        let (na, nb) = match op {
            BinOp::Lt => (
                ra.intersect(IntRange { lo: i128::MIN, hi: rb.hi.saturating_sub(1) }),
                rb.intersect(IntRange { lo: ra.lo.saturating_add(1), hi: i128::MAX }),
            ),
            BinOp::Le => (
                ra.intersect(IntRange { lo: i128::MIN, hi: rb.hi }),
                rb.intersect(IntRange { lo: ra.lo, hi: i128::MAX }),
            ),
            BinOp::Gt => (
                ra.intersect(IntRange { lo: rb.lo.saturating_add(1), hi: i128::MAX }),
                rb.intersect(IntRange { lo: i128::MIN, hi: ra.hi.saturating_sub(1) }),
            ),
            BinOp::Ge => (
                ra.intersect(IntRange { lo: rb.lo, hi: i128::MAX }),
                rb.intersect(IntRange { lo: i128::MIN, hi: ra.hi }),
            ),
            BinOp::Eq => {
                let both = ra.intersect(rb);
                (both, both)
            }
            BinOp::Ne => {
                let na = if rb.lo == rb.hi {
                    if ra.lo == rb.lo {
                        ra.intersect(IntRange { lo: ra.lo.saturating_add(1), hi: ra.hi })
                    } else if ra.hi == rb.hi {
                        ra.intersect(IntRange { lo: ra.lo, hi: ra.hi.saturating_sub(1) })
                    } else {
                        Some(ra)
                    }
                } else {
                    Some(ra)
                };
                (na, Some(rb))
            }
            _ => return,
        };
        if let Some(r) = na {
            self.ints.insert(a, r);
            if let Some(&l) = self.load_of.get(&a) {
                self.local_ints.insert(l, r);
            }
        }
        if let Some(r) = nb {
            self.ints.insert(b, r);
            if let Some(&l) = self.load_of.get(&b) {
                self.local_ints.insert(l, r);
            }
        }
    }

    fn narrow_float_cmp(&mut self, op: BinOp, a: ValueId, b: ValueId) {
        let fa = self.floats.get(&a).copied();
        let fb = self.floats.get(&b).copied();
        if let Some((blo, bhi)) = fb {
            match op {
                BinOp::Gt | BinOp::Ge => {
                    if blo >= 0.0 {
                        self.nonneg_floats.insert(a);
                    }
                    if let Some((alo, ahi)) = fa
                        && blo <= ahi
                    {
                        self.floats.insert(a, (alo.max(blo), ahi));
                    }
                }
                BinOp::Lt | BinOp::Le => {
                    if let Some((alo, ahi)) = fa
                        && alo <= bhi
                    {
                        self.floats.insert(a, (alo, ahi.min(bhi)));
                    }
                }
                _ => {}
            }
        }
        if let Some((alo, ahi)) = fa {
            match op {
                BinOp::Lt | BinOp::Le => {
                    if alo >= 0.0 {
                        self.nonneg_floats.insert(b);
                    }
                    if let Some((blo, bhi)) = fb
                        && alo <= bhi
                    {
                        self.floats.insert(b, (blo.max(alo), bhi));
                    }
                }
                BinOp::Gt | BinOp::Ge => {
                    if let Some((blo, bhi)) = fb
                        && blo <= ahi
                    {
                        self.floats.insert(b, (blo, bhi.min(ahi)));
                    }
                }
                _ => {}
            }
        }
    }

    /// Detects canonical ascending `for`-loop induction variables (`local = c0; loop { if !(local < bound) { break } ... continuing { local = local + 1 } }`)
    /// and returns `local`'s range at the top of the body, where the exit test reads it:
    /// `IntRange { lo: init.lo, hi: bound_hi }` (or `init.hi`, if more), which holds the value
    /// that ends the loop. Past the test, the test's own narrowing gives `bound_hi - 1`. (A range
    /// of `bound_hi - 1` at the top settled the test itself and erased the loop's exit.)
    pub fn detect_loop_induction(
        &self,
        types: &Types,
        f: &Function,
        body: &[Stmt],
        continuing: &[Stmt],
    ) -> Option<(LocalId, IntRange, Option<PlaceRoot>)> {
        // Continuing block must have a single store `l = l + 1` (plus pure `let`s).
        let (inc_local, step) = find_simple_increment(continuing)?;
        if step != 1 {
            return None;
        }
        // Check `inc_local` is not stored anywhere inside `body`.
        let mut modified_in_body = false;
        visit::walk(body, &mut |s| match s {
            Stmt::Store(Place { root: PlaceRoot::Local(l), .. }, _) if *l == inc_local => {
                modified_in_body = true;
            }
            _ => {}
        });
        if modified_in_body {
            return None;
        }
        let init = *self.local_ints.get(&inc_local)?;
        if init.lo < 0 {
            return None;
        }
        // Look at the top of `body` for the canonical guard `let v = Load(inc_local); let cmp = v < bound; if !cmp { break }`
        // or `let cmp = v >= bound; if cmp { break }`.
        let mut env = self.clone();
        // Before the guard, `inc_local` is at least `init.lo` (upper bound unknown yet).
        let scalar = types.as_scalar(f.locals[inc_local.index()].ty)?;
        let lim = IntRange::of_scalar(scalar)?;
        env.local_ints.insert(inc_local, IntRange { lo: init.lo, hi: lim.hi });
        for s in body {
            match s {
                Stmt::Let(v, e) => env.record_let(types, f, *v, e),
                Stmt::If { cond, then, else_ }
                    if matches!(then.last(), Some(Stmt::Break)) && else_.is_empty() =>
                {
                    // Guard exits when `cond` is true, so the rest of `body` runs when `!cond`.
                    env.narrow_cond(types, f, *cond, false);
                    let r = *env.local_ints.get(&inc_local)?;
                    // Check if any value loaded from `inc_local` gained a `< ArrayLength(root)` fact.
                    let mut lt_root = None;
                    for &(val, ref root) in &env.lt_len {
                        if env.load_of.get(&val) == Some(&inc_local) {
                            lt_root = Some(root.clone());
                            break;
                        }
                    }
                    if r.lo >= init.lo && r.hi < lim.hi {
                        let top = IntRange { lo: init.lo, hi: (r.hi + 1).max(init.hi) };
                        return Some((inc_local, top, lt_root));
                    }
                    return None;
                }
                _ => return None,
            }
        }
        None
    }
}

/// Whether every control-flow path through `b` exits the enclosing block (via `Return`, `Break`,
/// `Continue`, or `Trap`). Used for post-dominator narrowing after `if bad { return }`.
pub fn always_exits(b: &[Stmt]) -> bool {
    for s in b {
        match s {
            Stmt::Return(_) | Stmt::Break | Stmt::Continue | Stmt::Trap => return true,
            Stmt::If { then, else_, .. } if always_exits(then) && always_exits(else_) => {
                return true;
            }
            _ => {}
        }
    }
    false
}

fn find_simple_increment(continuing: &[Stmt]) -> Option<(LocalId, i128)> {
    let mut loads: HashMap<ValueId, LocalId> = HashMap::new();
    let mut consts: HashMap<ValueId, i128> = HashMap::new();
    let mut adds: HashMap<ValueId, (ValueId, ValueId)> = HashMap::new();
    let mut found: Option<(LocalId, i128)> = None;
    for s in continuing {
        match s {
            Stmt::Let(v, Expr::Load(Place { root: PlaceRoot::Local(l), path }))
                if path.is_empty() =>
            {
                loads.insert(*v, *l);
            }
            Stmt::Let(v, Expr::Const(c)) => {
                if let Some(n) = int_const_val(c) {
                    consts.insert(*v, n);
                }
            }
            Stmt::Let(v, Expr::Binary(BinOp::Add, a, b)) => {
                adds.insert(*v, (*a, *b));
            }
            Stmt::Store(Place { root: PlaceRoot::Local(l), path }, val) if path.is_empty() => {
                if found.is_some() {
                    return None;
                }
                let &(a, b) = adds.get(val)?;
                if loads.get(&a) == Some(l) && consts.contains_key(&b) {
                    found = Some((*l, consts[&b]));
                } else if loads.get(&b) == Some(l) && consts.contains_key(&a) {
                    found = Some((*l, consts[&a]));
                } else {
                    return None;
                }
            }
            Stmt::Let(..) => {}
            _ => return None,
        }
    }
    found
}

fn int_const_val(c: &Const) -> Option<i128> {
    Some(match *c {
        Const::I32(x) => i128::from(x),
        Const::U32(x) => i128::from(x),
        Const::I64(x) => i128::from(x),
        Const::U64(x) => i128::from(x),
        Const::Small(_, x) => i128::from(x),
        _ => return None,
    })
}

fn negate_cmp(op: BinOp) -> BinOp {
    match op {
        BinOp::Lt => BinOp::Ge,
        BinOp::Le => BinOp::Gt,
        BinOp::Gt => BinOp::Le,
        BinOp::Ge => BinOp::Lt,
        BinOp::Eq => BinOp::Ne,
        BinOp::Ne => BinOp::Eq,
        other => other,
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

    #[test]
    fn a_bounded_function_of_an_unbounded_float_has_no_range() {
        // `sin`, `cos`, `atan`, `sign`, `fract` and `smoothstep` of a load (which may be a NaN or an
        // infinity) have no range, so no guard after them is erased; of a clamp, they do.
        let mut f = Function::new("f", Vec::new(), None);
        let v = ValueId;
        let l = LocalId(0);
        let mut body = vec![
            Stmt::Let(v(0), Expr::Load(Place::local(l))),
            Stmt::Let(v(1), Expr::Const(Const::F32(-4.0))),
            Stmt::Let(v(2), Expr::Const(Const::F32(4.0))),
            Stmt::Let(v(3), Expr::Builtin(Builtin::Clamp, vec![v(0), v(1), v(2)])),
        ];
        let ops = [Builtin::Sin, Builtin::Cos, Builtin::Atan, Builtin::Sign, Builtin::Fract];
        for (k, b) in ops.iter().enumerate() {
            let k = k as u32;
            body.push(Stmt::Let(v(10 + k), Expr::Builtin(*b, vec![v(0)])));
            body.push(Stmt::Let(v(20 + k), Expr::Builtin(*b, vec![v(3)])));
        }
        body.push(Stmt::Let(v(30), Expr::Builtin(Builtin::Smoothstep, vec![v(1), v(2), v(0)])));
        body.push(Stmt::Let(v(31), Expr::Builtin(Builtin::Smoothstep, vec![v(1), v(2), v(3)])));
        body.push(Stmt::Let(v(32), Expr::Builtin(Builtin::Step, vec![v(1), v(0)])));
        f.body = body;
        let mut r = Ranges::new(&f);
        for k in 0..ops.len() as u32 {
            assert!(r.of(v(10 + k)).is_none(), "{:?} of a load has a range", ops[k as usize]);
            assert!(r.within(v(20 + k), -1.58, 1.58), "{:?} of a clamp has none", ops[k as usize]);
        }
        assert!(r.of(v(30)).is_none());
        assert!(r.within(v(31), 0.0, 1.0));
        assert!(r.within(v(32), 0.0, 1.0));
    }

    #[test]
    fn flow_env_proves_bitmask_modulo_and_storage_guard_in_bounds() {
        let mut m = Module::default();
        let u32_ty = m.types.u32();
        let bool_ty = m.types.bool();
        let buf = m.add_resource(Resource {
            name: "buf".into(),
            binding: 0,
            kind: ResourceKind::StorageReadWrite,
            ty: u32_ty,
        });
        let mut f = Function::new("k", vec![Param::value("id", u32_ty)], None);
        let mut body = Vec::new();
        let id = f.let_(&mut body, u32_ty, Expr::Param(0));
        let mask = f.let_(&mut body, u32_ty, Expr::Const(Const::U32(7)));
        let masked = f.let_(&mut body, u32_ty, Expr::Binary(BinOp::BitAnd, id, mask));
        let len =
            f.let_(&mut body, u32_ty, Expr::ArrayLength(Place::root(PlaceRoot::Resource(buf))));
        let oob = f.let_(&mut body, bool_ty, Expr::Binary(BinOp::Ge, id, len));
        let mut env = FlowEnv::new();
        for s in &body {
            if let Stmt::Let(v, e) = s {
                env.record_let(&m.types, &f, *v, e);
            }
        }
        assert!(env.in_const_bounds(&m.types, &f, masked, 8));
        assert!(!env.in_const_bounds(&m.types, &f, masked, 7));
        assert!(!env.in_storage_bounds(&m.types, &f, id, &PlaceRoot::Resource(buf)));
        // After `if id >= len { return }`, `oob` is false => `id < len`!
        env.narrow_cond(&m.types, &f, oob, false);
        assert!(env.in_storage_bounds(&m.types, &f, id, &PlaceRoot::Resource(buf)));
    }
}
