//! A strict module's float arithmetic (language.md §11.3, `@strict`): what a strict GPU kernel
//! computes is what the CPU computes from the same source, operation for operation. The shader
//! writers write a strict pipeline with no contraction and, on Metal, safe math, so each `+`,
//! `-`, `*` and `/` of floats is one IEEE 754 rounding there, as on the CPU (with the GPU's
//! subnormals flushed: `toolchain/host`'s `the_gpu_rounds_the_safe_operations_as_the_cpu_does`
//! holds each machine to the model). What's left is the built-in functions a driver computes
//! its own way: [`lower_strict`] writes each as the CPU's WASM back end computes it, from those
//! operations, so no driver's `mix`, `dot`, `smoothstep`, `min` or `clamp` decides a bit. The
//! built-ins it keeps are the exact ones (`abs`, `floor`, `ceil`, `trunc`, `round` to even, the
//! integer ones), `sqrt`, and the transcendentals, which lowering then replaces by calls to
//! std's (`std::math`, the CPU's own code).

use crate::*;

/// Writes every float built-in of every function of `m` that a driver computes its own way as
/// the CPU computes it (`toolchain/wasm`'s `scalar_builtin` and `builtin`), and every product
/// with a matrix as the CPU sums it.
pub fn lower_strict(m: &mut Module) {
    for i in 0..m.functions.len() {
        let mut f = std::mem::take(&mut m.functions[i]);
        let mut body = std::mem::take(&mut f.body);
        visit::expand(&mut body, &mut |s, out| match s {
            Stmt::Let(v, e) => {
                let mut em = Emit { m, f: &mut f, out };
                if !em.strict_let(v, &e) {
                    em.out.push(Stmt::Let(v, e));
                }
            }
            s => out.push(s),
        });
        f.body = body;
        m.functions[i] = f;
    }
}

/// Whether the built-in `b` is computed exactly by every GPU and the CPU alike, or is left for
/// lowering (`sqrt` and the transcendentals): kept as it is in a strict module.
pub fn kept_in_strict(b: Builtin) -> bool {
    use Builtin as B;
    matches!(
        b,
        B::Abs
            | B::Floor
            | B::Ceil
            | B::Trunc
            | B::Round
            | B::Sqrt
            | B::Sin
            | B::Cos
            | B::Tan
            | B::Asin
            | B::Acos
            | B::Atan
            | B::Atan2
            | B::Exp
            | B::Exp2
            | B::Log
            | B::Log2
            | B::Sinh
            | B::Cosh
            | B::Tanh
            | B::Pow
            | B::AllEqual
            | B::CountOnes
            | B::LeadingZeros
            | B::TrailingZeros
            | B::Dpdx
            | B::Dpdy
            | B::Fwidth
    )
}

struct Emit<'a> {
    m: &'a mut Module,
    f: &'a mut Function,
    out: &'a mut Block,
}

impl Emit<'_> {
    fn let_(&mut self, t: TypeId, e: Expr) -> ValueId {
        self.f.let_(self.out, t, e)
    }

    /// Makes `v` the value `r` that the last statement defines (each expansion ends with it).
    fn define(&mut self, v: ValueId, r: ValueId) {
        match self.out.last_mut() {
            Some(Stmt::Let(d, _)) if *d == r => *d = v,
            _ => unreachable!("an expansion ends by defining its value"),
        }
    }

    fn ty(&self, v: ValueId) -> TypeId {
        self.f.value_ty(v)
    }

    fn f32c(&mut self, x: f32) -> ValueId {
        let t = self.m.types.f32();
        self.let_(t, Expr::Const(Const::F32(x)))
    }

    /// Component `i` of `x`, or `x` itself if it's a scalar.
    fn comp(&mut self, x: ValueId, i: u32) -> ValueId {
        let t = self.ty(x);
        match *self.m.types.get(t) {
            TypeDef::Vector(s, _) => {
                let ct = self.m.types.scalar(s);
                self.let_(ct, Expr::Extract(x, i))
            }
            _ => x,
        }
    }

    fn bin(&mut self, op: BinOp, a: ValueId, b: ValueId) -> ValueId {
        let t = self.m.types.f32();
        self.let_(t, Expr::Binary(op, a, b))
    }

    fn cmp(&mut self, op: BinOp, a: ValueId, b: ValueId) -> ValueId {
        let t = self.m.types.bool();
        self.let_(t, Expr::Binary(op, a, b))
    }

    fn select(&mut self, cond: ValueId, if_true: ValueId, if_false: ValueId) -> ValueId {
        let t = self.ty(if_true);
        self.let_(t, Expr::Select { cond, if_true, if_false })
    }

    /// `min(a, b)` or `max(a, b)` of two f32s (§11.4): `b` where it's less (more) than `a`, else
    /// `a`, as the CPU computes it: a compare and a select.
    fn min_max(&mut self, max: bool, a: ValueId, b: ValueId) -> ValueId {
        let picks_b = self.cmp(if max { BinOp::Gt } else { BinOp::Lt }, b, a);
        self.select(picks_b, b, a)
    }

    /// The f32 `b` of the scalars `xs` (one component of each argument), as the CPU computes
    /// it, or `None` to keep it.
    fn scalar(&mut self, b: Builtin, xs: &[ValueId]) -> Option<ValueId> {
        use Builtin as B;
        Some(match b {
            B::InverseSqrt => {
                let f = self.m.types.f32();
                let s = self.let_(f, Expr::Builtin(B::Sqrt, vec![xs[0]]));
                let one = self.f32c(1.0);
                self.bin(BinOp::Div, one, s)
            }
            B::Fract => {
                let f = self.m.types.f32();
                let fl = self.let_(f, Expr::Builtin(B::Floor, vec![xs[0]]));
                self.bin(BinOp::Sub, xs[0], fl)
            }
            B::Sign => {
                let (zero, one, minus) = (self.f32c(0.0), self.f32c(1.0), self.f32c(-1.0));
                let neg = self.cmp(BinOp::Lt, xs[0], zero);
                let inner = self.select(neg, minus, xs[0]);
                let pos = self.cmp(BinOp::Gt, xs[0], zero);
                self.select(pos, one, inner)
            }
            B::Min => self.min_max(false, xs[0], xs[1]),
            B::Max => self.min_max(true, xs[0], xs[1]),
            B::Clamp => {
                let lo = self.min_max(true, xs[0], xs[1]);
                self.min_max(false, lo, xs[2])
            }
            B::Saturate => {
                let (zero, one) = (self.f32c(0.0), self.f32c(1.0));
                let lo = self.min_max(true, xs[0], zero);
                self.min_max(false, lo, one)
            }
            B::Mix => {
                // e1 * (1 - e3) + e2 * e3.
                let one = self.f32c(1.0);
                let w = self.bin(BinOp::Sub, one, xs[2]);
                let p = self.bin(BinOp::Mul, xs[0], w);
                let q = self.bin(BinOp::Mul, xs[1], xs[2]);
                self.bin(BinOp::Add, p, q)
            }
            B::Step => {
                let (zero, one) = (self.f32c(0.0), self.f32c(1.0));
                let le = self.cmp(BinOp::Le, xs[0], xs[1]);
                self.select(le, one, zero)
            }
            B::Smoothstep => {
                // t = clamp((x - e0) / (e1 - e0), 0, 1); t * t * (3 - 2t).
                let num = self.bin(BinOp::Sub, xs[2], xs[0]);
                let den = self.bin(BinOp::Sub, xs[1], xs[0]);
                let q = self.bin(BinOp::Div, num, den);
                let (zero, one) = (self.f32c(0.0), self.f32c(1.0));
                let lo = self.min_max(true, q, zero);
                let t = self.min_max(false, lo, one);
                let tt = self.bin(BinOp::Mul, t, t);
                let (three, two) = (self.f32c(3.0), self.f32c(2.0));
                let t2 = self.bin(BinOp::Mul, two, t);
                let k = self.bin(BinOp::Sub, three, t2);
                self.bin(BinOp::Mul, tt, k)
            }
            // A scalar's length and distance: an absolute value.
            B::Length => self.let_(self.ty(xs[0]), Expr::Builtin(B::Abs, vec![xs[0]])),
            B::Distance => {
                let d = self.bin(BinOp::Sub, xs[0], xs[1]);
                let f = self.m.types.f32();
                self.let_(f, Expr::Builtin(B::Abs, vec![d]))
            }
            _ => return None,
        })
    }

    /// The sum of the products of `n` pairs, in order, from the first (`from_zero`: from 0).
    fn sum_of_products(
        &mut self,
        n: u32,
        from_zero: bool,
        mut pair: impl FnMut(&mut Self, u32) -> (ValueId, ValueId),
    ) -> ValueId {
        let mut acc = if from_zero { Some(self.f32c(0.0)) } else { None };
        for k in 0..n {
            let (x, y) = pair(self, k);
            let p = self.bin(BinOp::Mul, x, y);
            acc = Some(match acc {
                None => p,
                Some(a) => self.bin(BinOp::Add, a, p),
            });
        }
        acc.expect("a sum of something")
    }

    /// Rewrites `let v = e` if it's a float built-in or a matrix product a driver computes its
    /// own way; whether it did.
    fn strict_let(&mut self, v: ValueId, e: &Expr) -> bool {
        use Builtin as B;
        let vt = self.ty(v);
        match e {
            Expr::Builtin(b, args) if !kept_in_strict(*b) => {
                let arg0 = self.ty(args[0]);
                // Only floats: an integer's `min`, `clamp` or `abs` is exact everywhere.
                if self.m.types.element_scalar(arg0) != Some(Scalar::F32) {
                    return false;
                }
                let n = match *self.m.types.get(arg0) {
                    TypeDef::Vector(_, n) => n,
                    _ => 1,
                };
                let r = match b {
                    B::Dot => {
                        let (a, c) = (args[0], args[1]);
                        self.sum_of_products(u32::from(n), false, |em, k| {
                            (em.comp(a, k), em.comp(c, k))
                        })
                    }
                    B::Length | B::Distance | B::Normalize if n > 1 => {
                        let a = args[0];
                        let d = if *b == B::Distance {
                            let t = self.ty(a);
                            self.let_(t, Expr::Binary(BinOp::Sub, a, args[1]))
                        } else {
                            a
                        };
                        let ss = self.sum_of_products(u32::from(n), false, |em, k| {
                            let c = em.comp(d, k);
                            (c, c)
                        });
                        let f = self.m.types.f32();
                        let len = self.let_(f, Expr::Builtin(B::Sqrt, vec![ss]));
                        if *b == B::Normalize {
                            // x · (1 / |x|): one division, as the CPU's.
                            let one = self.f32c(1.0);
                            let r = self.bin(BinOp::Div, one, len);
                            let parts: Vec<ValueId> = (0..u32::from(n))
                                .map(|k| {
                                    let c = self.comp(a, k);
                                    self.bin(BinOp::Mul, c, r)
                                })
                                .collect();
                            self.let_(vt, Expr::Construct(vt, parts))
                        } else {
                            len
                        }
                    }
                    B::Normalize => {
                        // A scalar: x / |x|.
                        let f = self.m.types.f32();
                        let len = self.let_(f, Expr::Builtin(B::Abs, vec![args[0]]));
                        self.bin(BinOp::Div, args[0], len)
                    }
                    B::Cross => {
                        let (a, c) = (args[0], args[1]);
                        let parts: Vec<ValueId> = [(1u32, 2u32), (2, 0), (0, 1)]
                            .into_iter()
                            .map(|(j, k)| {
                                let (aj, ck, ak, cj) = (
                                    self.comp(a, j),
                                    self.comp(c, k),
                                    self.comp(a, k),
                                    self.comp(c, j),
                                );
                                let p = self.bin(BinOp::Mul, aj, ck);
                                let q = self.bin(BinOp::Mul, ak, cj);
                                self.bin(BinOp::Sub, p, q)
                            })
                            .collect();
                        self.let_(vt, Expr::Construct(vt, parts))
                    }
                    _ => {
                        // Elementwise: each component of each argument (a scalar argument to a
                        // vector's built-in is every component's).
                        let mut parts = Vec::new();
                        for k in 0..u32::from(n) {
                            let xs: Vec<ValueId> = args.iter().map(|&a| self.comp(a, k)).collect();
                            match self.scalar(*b, &xs) {
                                Some(p) => parts.push(p),
                                None => return false,
                            }
                        }
                        if n == 1 { parts[0] } else { self.let_(vt, Expr::Construct(vt, parts)) }
                    }
                };
                self.define(v, r);
                true
            }
            Expr::Binary(BinOp::Mul, a, b) => {
                let (ta, tb) = (self.ty(*a), self.ty(*b));
                let dims = |m: &Module, t: TypeId| match *m.types.get(t) {
                    TypeDef::Matrix(c, r) => Some((true, c, r)),
                    TypeDef::Vector(Scalar::F32, n) => Some((false, 1, n)),
                    _ => None,
                };
                let (Some(da), Some(db)) = (dims(self.m, ta), dims(self.m, tb)) else {
                    return false;
                };
                if !da.0 && !db.0 {
                    return false;
                }
                let elem = |em: &mut Self, x: ValueId, d: (bool, u8, u8), col: u32, row: u32| {
                    if d.0 {
                        let ct = em.m.types.vector(d.2);
                        let c = em.let_(ct, Expr::Extract(x, col));
                        em.comp(c, row)
                    } else {
                        em.comp(x, row)
                    }
                };
                let (av, bv) = (*a, *b);
                let r = match (da.0, db.0) {
                    // matrix × vector: r_i = Σ_j m[j][i] v[j], from zero.
                    (true, false) => {
                        let parts: Vec<ValueId> = (0..u32::from(da.2))
                            .map(|i| {
                                self.sum_of_products(u32::from(da.1), true, |em, j| {
                                    (elem(em, av, da, j, i), em.comp(bv, j))
                                })
                            })
                            .collect();
                        self.let_(vt, Expr::Construct(vt, parts))
                    }
                    // vector × matrix: r_j = Σ_i v[i] m[j][i], from zero.
                    (false, true) => {
                        let parts: Vec<ValueId> = (0..u32::from(db.1))
                            .map(|j| {
                                self.sum_of_products(u32::from(db.2), true, |em, i| {
                                    (em.comp(av, i), elem(em, bv, db, j, i))
                                })
                            })
                            .collect();
                        self.let_(vt, Expr::Construct(vt, parts))
                    }
                    // matrix × matrix: column j of the result is a × b[j].
                    _ => {
                        let cols: Vec<ValueId> = (0..u32::from(db.1))
                            .map(|j| {
                                let parts: Vec<ValueId> = (0..u32::from(da.2))
                                    .map(|i| {
                                        self.sum_of_products(u32::from(da.1), true, |em, k| {
                                            (elem(em, av, da, k, i), elem(em, bv, db, j, k))
                                        })
                                    })
                                    .collect();
                                let ct = self.m.types.vector(da.2);
                                self.let_(ct, Expr::Construct(ct, parts))
                            })
                            .collect();
                        self.let_(vt, Expr::Construct(vt, cols))
                    }
                };
                self.define(v, r);
                true
            }
            _ => false,
        }
    }
}
