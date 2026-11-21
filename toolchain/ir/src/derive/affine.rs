//! Affine arithmetic, first order with one error term (Stolfi and de Figueiredo, "Self-validated
//! numerical methods and applications", 1997; the reduced form is Messine's AF1). Over a box,
//! an active float is `c + l₁e₁ + … + lₙeₙ + err·e'`, each `e` in [-1, 1]: `eᵢ` the box's own
//! coordinates (the input is its centre plus its half-width on axis `i` times `eᵢ`), `e'` one
//! symbol for everything else. Sums and scalings are exact; a product keeps its operands' linear
//! parts and puts their cross term in `err`; a nonlinear function becomes its min-range linear
//! approximation over its operand's range (Chebyshev's, for the convex and concave pieces std's
//! fields use), the gap in `err`. Where the field is smooth over a box, `err` shrinks with the
//! box's square while the range shrinks with its width; and `x - x` is 0, which a plain interval
//! can't see.
//!
//! [`affine`] intersects the form's range with the plain interval's, end by end, so it is never
//! looser than the interval: it is what `std::derive::affine` gives.
//!
//! Floats and bools only. An active bool is what a box surely makes it: surely true, surely false,
//! or neither. A branch on one that is neither runs both sides and joins what they assign (the
//! side a box surely takes, else the hull, which keeps the two sides' mean slopes), with `return`s
//! first turned into stores (`single_exit`). `std::field::value_noise` is answered whole (its
//! value and gradient at the centre, a bound on its Hessian), and `std::field::smin` by its
//! closed form. An operation with no rule here takes the plain interval's rule for itself alone
//! (a function of just that operation, derived by [`interval`](super::interval)), and so does a
//! call whose callee can't be derived: the form loses its slopes there and goes on.
//!
//! Soundness as the target rounds: every operation adds to `err` a few ulps of its result's
//! magnitude (`|c| + Σ|lᵢ| + err`), as many as its own arithmetic and the point code's rounding of
//! the same operation take together (more on the GPU, by WGSL's bounds), and a subnormal. The
//! input's centre carries its own rounding in its error. An unbounded error is held at `BIG`, so
//! no `∞ × 0` makes a NaN. A form that reaches past `CAP` stands for no bound: what a product, a
//! quotient or a builtin makes of it is every number (a small factor doesn't bound it), a
//! comparison of it decides nothing, and as the answer it leaves the plain interval's ends, as a
//! NaN that gets through does.

use super::*;
use std::collections::{HashMap, HashSet};

type R<T> = Result<T>;

/// One ulp at 1, an `f32`'s.
const EPS: f64 = 1.0 / 8_388_608.0;
/// The unit roundoff, half of `EPS`: round to nearest is within it of the exact result.
const U: f64 = EPS / 2.0;
/// An error this big means every number: finite, so arithmetic on it makes no NaN.
const BIG: f64 = 1.0e30;
/// The most a radius is taken to be in a product's cross term, so that it stays under `BIG`.
const CAP: f64 = 1.0e15;
/// A bound on the norm of std's value noise's Hessian: its quintic fade's second derivative is
/// at most 5.774 and its first 1.875, between lattice values in [-1, 1], so each second
/// derivative along an axis is at most 11.55 and each mixed one 14.06, and the norm (at most the
/// Frobenius norm) is under 40.
const NOISE_HESSIAN: f64 = 40.0;
/// Value noise's own rounding inside a cell, absolute (its values are within 1), and its
/// derived gradient's, relative to the gradient's reach.
const NOISE_ROUNDING: f64 = 64.0 * EPS;

/// An affine value in the derived code: its centre, its slope on each axis, its error.
#[derive(Clone, Debug)]
struct A {
    c: ValueId,
    l: Vec<ValueId>,
    e: ValueId,
}

/// What a box makes an active bool: surely true, surely false (never both).
type Sure = (ValueId, ValueId);

/// A local's state across a branch: a float's form or a bool's certainty.
enum State {
    Float(A),
    Bool(Sure),
}

/// The affine form of `f` over a box, intersected with its plain interval: a function of `f`'s
/// captures and the box (`box_ty`) returning `interval_ty`. A function the form can't be
/// derived for gets the plain interval alone.
pub(super) fn affine(
    m: &mut Module,
    cache: &mut DeriveCache,
    f: FuncId,
    ncap: u32,
    target: Target,
    (box_ty, interval_ty): (TypeId, TypeId),
) -> R<FuncId> {
    let plain = super::interval::interval(m, cache, f, ncap, target, (box_ty, interval_ty), None)?;
    let func = m.functions[f.index()].clone_signature();
    let xi = ncap as usize;
    let x_ty = func.params[xi].ty;
    let n = match m.types.get(x_ty) {
        TypeDef::Scalar(Scalar::F32) => 1,
        TypeDef::Vector(Scalar::F32, k) => *k as usize,
        _ => return Ok(plain),
    };
    let f32 = m.types.f32();
    if func.ret != Some(f32) || func.ret_ref {
        return Ok(plain);
    }
    let mut mask = vec![0; func.params.len()];
    mask[xi] = all_bits(&m.types, x_ty);
    if returns_active(m, cache, f, mask.clone(), Mode::Interval) == 0 {
        return Ok(plain);
    }
    let d = match derive(m, cache, f, mask, n, target) {
        Ok(d) => d,
        Err(Error::NotDerivable(..) | Error::ActiveLoopExit(..)) => {
            return Ok(plain);
        }
        Err(e) => return Err(e),
    };
    let tiny = tiny(target);
    let mut params = func.params[..xi].to_vec();
    params.push(Param::value("over", box_ty));
    let mut g = Function::new(format!("{}_affine", func.name), params, Some(interval_ty));
    let mut body = Vec::new();
    let caps: Vec<Arg> = (0..xi as u32).map(|i| g.param_arg(i, &mut body)).collect();
    let over = g.let_(&mut body, box_ty, Expr::Param(xi as u32));
    let lo = g.let_(&mut body, x_ty, Expr::Extract(over, 0));
    let hi = g.let_(&mut body, x_ty, Expr::Extract(over, 1));
    let k = |g: &mut Function, body: &mut Block, x: f64| -> ValueId {
        let s = g.let_(body, f32, Expr::Const(Const::F32(x as f32)));
        if n == 1 { s } else { g.let_(body, x_ty, Expr::Splat(s, n as u8)) }
    };
    let half = k(&mut g, &mut body, 0.5);
    let hl = g.let_(&mut body, x_ty, Expr::Binary(BinOp::Mul, lo, half));
    let hh = g.let_(&mut body, x_ty, Expr::Binary(BinOp::Mul, hi, half));
    let c = g.let_(&mut body, x_ty, Expr::Binary(BinOp::Add, hl, hh));
    let h = g.let_(&mut body, x_ty, Expr::Binary(BinOp::Sub, hh, hl));
    let up = k(&mut g, &mut body, 1.0 + 2.0 * EPS);
    let h = g.let_(&mut body, x_ty, Expr::Binary(BinOp::Mul, h, up));
    // The centre is within an ulp of the middle: that ulp is its error.
    let ac = g.let_(&mut body, x_ty, Expr::Builtin(Builtin::Abs, vec![c]));
    let eps = k(&mut g, &mut body, EPS);
    let tv = k(&mut g, &mut body, tiny);
    let e = g.let_(&mut body, x_ty, Expr::Binary(BinOp::Mul, ac, eps));
    let e = g.let_(&mut body, x_ty, Expr::Binary(BinOp::Add, e, tv));
    let zero = g.let_(&mut body, f32, Expr::Const(Const::F32(0.0)));
    let by_ref = func.params[xi].by_ref;
    let pass = |g: &mut Function, body: &mut Block, v: ValueId| -> Arg {
        if by_ref {
            let l = g.new_local("x", x_ty);
            body.push(Stmt::Store(Place::local(l), v));
            Arg::Place(Place::local(l))
        } else {
            Arg::Value(v)
        }
    };
    let mut args = caps.clone();
    args.push(pass(&mut g, &mut body, c));
    for i in 0..n {
        let l = if n == 1 {
            h
        } else {
            let hi_ = g.let_(&mut body, f32, Expr::Extract(h, i as u32));
            let comps = (0..n).map(|j| if j == i { hi_ } else { zero }).collect();
            g.let_(&mut body, x_ty, Expr::Construct(x_ty, comps))
        };
        args.push(pass(&mut g, &mut body, l));
    }
    args.push(pass(&mut g, &mut body, e));
    let rty = m.functions[d.index()].ret.ok_or_else(|| Error::internal("no affine result"))?;
    let r = g.let_(&mut body, rty, Expr::Call(d, args));
    let rc = g.let_(&mut body, f32, Expr::Extract(r, 0));
    let mut rad = g.let_(&mut body, f32, Expr::Extract(r, 1 + n as u32));
    for i in 0..n {
        let li = g.let_(&mut body, f32, Expr::Extract(r, 1 + i as u32));
        let a = g.let_(&mut body, f32, Expr::Builtin(Builtin::Abs, vec![li]));
        rad = g.let_(&mut body, f32, Expr::Binary(BinOp::Add, rad, a));
    }
    // The radius's own sum and the ends' subtraction, rounded: a few ulps more.
    let arc = g.let_(&mut body, f32, Expr::Builtin(Builtin::Abs, vec![rc]));
    let mg = g.let_(&mut body, f32, Expr::Binary(BinOp::Add, arc, rad));
    let kk = g.let_(&mut body, f32, Expr::Const(Const::F32((2.0 * (n as f64 + 2.0) * EPS) as f32)));
    let grow = g.let_(&mut body, f32, Expr::Binary(BinOp::Mul, mg, kk));
    let rad = g.let_(&mut body, f32, Expr::Binary(BinOp::Add, rad, grow));
    let t1 = g.let_(&mut body, f32, Expr::Const(Const::F32(tiny as f32)));
    let rad = g.let_(&mut body, f32, Expr::Binary(BinOp::Add, rad, t1));
    let alo = g.let_(&mut body, f32, Expr::Binary(BinOp::Sub, rc, rad));
    let ahi = g.let_(&mut body, f32, Expr::Binary(BinOp::Add, rc, rad));
    let mut pargs = caps;
    pargs.push(Arg::Value(over));
    let p = g.let_(&mut body, interval_ty, Expr::Call(plain, pargs));
    let plo = g.let_(&mut body, f32, Expr::Extract(p, 0));
    let phi = g.let_(&mut body, f32, Expr::Extract(p, 1));
    // The tighter end of the two; a NaN end of the form's leaves the plain one, and so does a
    // form that reaches past `CAP`, which stands for no bound (`Af::within`).
    let b = m.types.bool();
    let cap = g.let_(&mut body, f32, Expr::Const(Const::F32(CAP as f32)));
    let fine = g.let_(&mut body, b, Expr::Binary(BinOp::Le, mg, cap));
    let lo_ok = g.let_(&mut body, b, Expr::Binary(BinOp::Gt, alo, plo));
    let lo_ok = g.let_(&mut body, b, Expr::Binary(BinOp::And, lo_ok, fine));
    let hi_ok = g.let_(&mut body, b, Expr::Binary(BinOp::Lt, ahi, phi));
    let hi_ok = g.let_(&mut body, b, Expr::Binary(BinOp::And, hi_ok, fine));
    let rlo = g.let_(&mut body, f32, Expr::Select { cond: lo_ok, if_true: alo, if_false: plo });
    let rhi = g.let_(&mut body, f32, Expr::Select { cond: hi_ok, if_true: ahi, if_false: phi });
    let out = g.let_(&mut body, interval_ty, Expr::Construct(interval_ty, vec![rlo, rhi]));
    body.push(Stmt::Return(Some(out)));
    g.body = body;
    Ok(m.add_function(g))
}

/// The affine version of `f` with active parameters `mask`, over `n` axes. Each active
/// parameter takes its centre in its own slot, followed (after the originals) by its `n` slopes
/// and its error, each passed as the parameter is; an active result is `{ c, l₀ … lₙ₋₁, e }`.
fn derive(
    m: &mut Module,
    cache: &mut DeriveCache,
    f: FuncId,
    mask: Vec<Bits>,
    n: usize,
    target: Target,
) -> R<FuncId> {
    let key = (f, mask.clone(), n as u8);
    if let Some(&d) = cache.affine.get(&key) {
        return Ok(d);
    }
    if cache.affine_open.contains(&key) {
        return Err(Error::not_derivable(
            "a recursive call: an affine form runs both sides of a branch",
        ));
    }
    let orig = m.functions[f.index()].clone();
    no_gpu_or_heap_work(&orig.body)?;
    let returns = returns_active(m, cache, f, mask.clone(), Mode::Interval) != 0;
    let (func, result, act) = one_exit(m, cache, f, orig, &mask);
    if act.active_loop_exit.is_some() {
        return Err(Error::not_derivable("a loop whose exit depends on the input"));
    }
    no_heap_store(&act)?;
    no_write_through_mut(&func, &act, &mask, "an affine form")?;
    let bool_ty = m.types.bool();
    let mut nf = func.clone_signature();
    nf.name = format!("{}_a{n}", func.name);
    let mut pparts: Vec<Vec<u32>> = vec![Vec::new(); func.params.len()];
    for (i, p) in func.params.iter().enumerate() {
        if mask[i] != 0 {
            if !floaty(&m.types, p.ty) {
                return Err(Error::not_derivable("an active parameter that isn't made of floats"));
            }
            for k in 0..=n {
                pparts[i].push(nf.params.len() as u32);
                let name = if k < n { format!("{}_l{k}", p.name) } else { format!("{}_e", p.name) };
                // A `mut` parameter's slopes and error are written with it.
                nf.params.push(Param { name, ty: p.ty, by_ref: p.by_ref, mutable: p.mutable });
            }
        }
    }
    let mut lparts: Vec<Vec<LocalId>> = vec![Vec::new(); func.locals.len()];
    for (l, decl) in func.locals.iter().enumerate() {
        if act.locals[l] == 0 {
            continue;
        }
        if decl.ty == bool_ty {
            for s in ["sure", "never"] {
                lparts[l].push(nf.new_local(format!("{}_{s}", decl.name), bool_ty));
            }
        } else if floaty(&m.types, decl.ty) {
            for k in 0..=n {
                lparts[l].push(nf.new_local(format!("{}_a{k}", decl.name), decl.ty));
            }
        } else {
            return Err(Error::not_derivable("an active local that isn't a float or a bool"));
        }
    }
    if returns {
        let r = func.ret.ok_or_else(|| Error::internal("a missing result type"))?;
        if !floaty(&m.types, r) || func.ret_ref {
            return Err(Error::not_derivable("an active result that isn't made of floats"));
        }
        let mut fields = vec![("c".to_string(), r)];
        for k in 0..n {
            fields.push((format!("l{k}"), r));
        }
        fields.push(("e".to_string(), r));
        nf.ret_ref = false;
        nf.ret = Some(m.types.intern(TypeDef::Struct {
            name: format!("affine{n}<{}>", m.types.display(r)),
            fields,
        }));
    }
    cache.affine_open.push(key.clone());
    let ptrs = Pointers::of(m, &func);
    let mut b = Af {
        m,
        cache,
        f: &func,
        nf,
        act,
        n,
        target,
        bool_ty,
        pparts,
        lparts,
        forms: HashMap::new(),
        sure: HashMap::new(),
        zeros: HashSet::new(),
        returns,
        speculative: 0,
        ptrs,
    };
    let built = (|| -> R<Block> {
        let mut body = Vec::new();
        if let Some(r) = result.filter(|r| !b.lparts[r.index()].is_empty()) {
            // The result starts empty, so a side that never returns adds nothing to the hull.
            let ty = b.f.locals[r.index()].ty;
            let e = b.empty(&mut body, ty);
            b.store_form(&mut body, &Place::local(r), &e)?;
        }
        b.block_into(&func.body, &mut body)?;
        Ok(body)
    })();
    if let Some(i) = b.cache.affine_open.iter().rposition(|k| *k == key) {
        b.cache.affine_open.remove(i);
    }
    let mut nf = b.nf;
    nf.body = built?;
    let id = m.add_function(nf);
    cache.affine.insert(key, id);
    Ok(id)
}

/// Made of floats: a float, or a struct or array of what is (a pair a function returns, a stack
/// a loop indexes). A form of one is the aggregate of its leaves' forms: built, taken apart, kept
/// and passed whole; arithmetic and joins are on floats alone.
fn floaty(types: &Types, t: TypeId) -> bool {
    match types.get(t) {
        TypeDef::Struct { fields, .. } => {
            !fields.is_empty() && fields.iter().all(|(_, f)| floaty(types, *f))
        }
        &TypeDef::Array(e, _) => floaty(types, e),
        _ => is_float(types, t),
    }
}

fn is_float(types: &Types, t: TypeId) -> bool {
    matches!(types.get(t), TypeDef::Scalar(Scalar::F32) | TypeDef::Vector(Scalar::F32, _))
}

struct Af<'a> {
    m: &'a mut Module,
    cache: &'a mut DeriveCache,
    f: &'a Function,
    nf: Function,
    act: Activity,
    n: usize,
    target: Target,
    bool_ty: TypeId,
    /// An active parameter's slopes' and error's parameters.
    pparts: Vec<Vec<u32>>,
    /// An active local's: a float's slopes and error, a bool's `sure` and `never`.
    lparts: Vec<Vec<LocalId>>,
    forms: HashMap<ValueId, A>,
    sure: HashMap<ValueId, Sure>,
    /// The zeros `exact` makes: a form whose slopes and error are one of them is exactly its
    /// centre, a value that doesn't depend on the input.
    zeros: HashSet<ValueId>,
    returns: bool,
    /// How many branches on the input the code being derived is under.
    speculative: u32,
    /// Where each pointer may point: a write through one into an active place would change its
    /// centre and leave its slopes and error behind.
    ptrs: Pointers,
}

impl Builder for Af<'_> {
    fn emit(&mut self, out: &mut Block, ty: TypeId, e: Expr) -> ValueId {
        self.nf.let_(out, ty, e)
    }
    fn types(&self) -> &Types {
        &self.m.types
    }
    fn ty(&self, v: ValueId) -> TypeId {
        self.nf.value_ty(v)
    }
}

/// The least positive float a target keeps: the GPU flushes subnormals to zero.
fn tiny(target: Target) -> f64 {
    if target == Target::Gpu { 1.2e-38 } else { 1.0e-45 }
}

impl Af<'_> {
    fn gpu(&self) -> bool {
        self.target == Target::Gpu
    }

    fn tiny(&self) -> f64 {
        tiny(self.target)
    }

    /// `(cpu, gpu)` ulps as an error factor on this target.
    fn ulps(&self, (cpu, gpu): (f64, f64)) -> f64 {
        (if self.gpu() { gpu } else { cpu }) * EPS
    }

    /// A float constant of type `ty` (splatted for a vector).
    fn k(&mut self, out: &mut Block, ty: TypeId, x: f64) -> ValueId {
        let f = self.m.types.f32();
        let s = self.emit(out, f, Expr::Const(Const::F32(x as f32)));
        self.coerce(out, s, ty)
    }

    fn bconst(&mut self, out: &mut Block, b: bool) -> ValueId {
        let bt = self.bool_ty;
        self.emit(out, bt, Expr::Const(Const::Bool(b)))
    }

    /// The bool type of `ty`'s shape.
    fn bool_of(&mut self, ty: TypeId) -> TypeId {
        match self.m.types.get(ty) {
            &TypeDef::Vector(_, k) => self.m.types.intern(TypeDef::Vector(Scalar::Bool, k)),
            _ => self.bool_ty,
        }
    }

    fn cmp(&mut self, out: &mut Block, op: BinOp, a: ValueId, b: ValueId) -> ValueId {
        let ty = self.ty(a);
        let bt = self.bool_of(ty);
        let b = self.coerce(out, b, ty);
        self.emit(out, bt, Expr::Binary(op, a, b))
    }

    fn sel(&mut self, out: &mut Block, cond: ValueId, t: ValueId, f: ValueId) -> ValueId {
        let ty = self.ty(t);
        self.emit(out, ty, Expr::Select { cond, if_true: t, if_false: f })
    }

    fn zero(&mut self, out: &mut Block, ty: TypeId) -> ValueId {
        self.emit(out, ty, Expr::Zero(ty))
    }

    /// A vector type's length, or `None` for a scalar.
    fn width(&self, ty: TypeId) -> Option<u8> {
        match self.m.types.get(ty) {
            &TypeDef::Vector(_, k) => Some(k),
            _ => None,
        }
    }

    /// A vector form's components, each a scalar form.
    fn split(&mut self, out: &mut Block, a: &A, k: u8) -> Vec<A> {
        let f = self.m.types.f32();
        (0..k as u32)
            .map(|j| {
                let c = self.emit(out, f, Expr::Extract(a.c, j));
                let l = (0..self.n).map(|i| self.emit(out, f, Expr::Extract(a.l[i], j))).collect();
                let e = self.emit(out, f, Expr::Extract(a.e, j));
                A { c, l, e }
            })
            .collect()
    }

    /// A vector form from its components'.
    fn gather(&mut self, out: &mut Block, parts: &[A], ty: TypeId) -> A {
        let c = self.emit(out, ty, Expr::Construct(ty, parts.iter().map(|a| a.c).collect()));
        let l = (0..self.n)
            .map(|i| {
                self.emit(out, ty, Expr::Construct(ty, parts.iter().map(|a| a.l[i]).collect()))
            })
            .collect();
        let e = self.emit(out, ty, Expr::Construct(ty, parts.iter().map(|a| a.e).collect()));
        A { c, l, e }
    }

    /// A rule that chooses (a select takes one `bool`) applied to a vector component by
    /// component; `None` for scalars, which it applies to itself.
    fn per_comp(
        &mut self,
        out: &mut Block,
        args: &[&A],
        ty: TypeId,
        rule: fn(&mut Self, &mut Block, &[A], TypeId) -> A,
    ) -> Option<A> {
        let k = self.width(ty)?;
        let f = self.m.types.f32();
        let split: Vec<Vec<A>> = args.iter().map(|a| self.split(out, a, k)).collect();
        let parts: Vec<A> = (0..k as usize)
            .map(|j| {
                let comps: Vec<A> = split.iter().map(|s| s[j].clone()).collect();
                rule(self, out, &comps, f)
            })
            .collect();
        Some(self.gather(out, &parts, ty))
    }

    /// An inactive value as an exact form.
    fn exact(&mut self, out: &mut Block, v: ValueId) -> A {
        let ty = self.ty(v);
        let z = self.zero(out, ty);
        self.zeros.insert(z);
        A { c: v, l: vec![z; self.n], e: z }
    }

    /// Whether `a` is exactly its centre (`exact`'s): a scaling or a shift by it, not a product
    /// or a sum of forms.
    fn is_exact(&self, a: &A) -> bool {
        self.zeros.contains(&a.e) && a.l.iter().all(|x| *x == a.e)
    }

    /// The empty form (its range runs from `BIG` down to `-BIG`): the hull's identity.
    fn empty(&mut self, out: &mut Block, ty: TypeId) -> A {
        let z = self.zero(out, ty);
        let e = self.k(out, ty, -BIG);
        A { c: z, l: vec![z; self.n], e }
    }

    /// `v`'s form: an active value's, or an inactive one's, exactly.
    fn of(&mut self, out: &mut Block, v: ValueId) -> A {
        match self.forms.get(&v) {
            Some(a) => a.clone(),
            None => self.exact(out, v),
        }
    }

    /// `v`'s form as `ty` (a scalar splat to a vector, where an operation mixes them).
    fn of_as(&mut self, out: &mut Block, v: ValueId, ty: TypeId) -> A {
        let a = self.of(out, v);
        self.widen_to(out, a, ty)
    }

    fn widen_to(&mut self, out: &mut Block, a: A, ty: TypeId) -> A {
        if self.ty(a.c) == ty {
            return a;
        }
        if self.is_exact(&a) {
            let c = self.coerce(out, a.c, ty);
            return self.exact(out, c);
        }
        let c = self.coerce(out, a.c, ty);
        let l = (0..self.n).map(|i| self.coerce(out, a.l[i], ty)).collect();
        let e = self.coerce(out, a.e, ty);
        A { c, l, e }
    }

    /// What a box makes `v`, a bool: an active one's certainty, an inactive one's value.
    fn sure_of(&mut self, out: &mut Block, v: ValueId) -> Sure {
        if let Some(&s) = self.sure.get(&v) {
            return s;
        }
        let bt = self.bool_ty;
        let not = self.emit(out, bt, Expr::Unary(UnOp::Not, v));
        (v, not)
    }

    /// Σ|lᵢ| + err.
    fn rad(&mut self, out: &mut Block, a: &A, ty: TypeId) -> ValueId {
        let mut r = a.e;
        for i in 0..self.n {
            let x = self.builtin(out, Builtin::Abs, vec![a.l[i]], ty);
            r = self.bin(out, BinOp::Add, r, x, ty);
        }
        r
    }

    /// `|c| + rad`: the most a value of the form's can be.
    fn magnitude(&mut self, out: &mut Block, a: &A, ty: TypeId) -> ValueId {
        let r = self.rad(out, a, ty);
        let ac = self.builtin(out, Builtin::Abs, vec![a.c], ty);
        self.bin(out, BinOp::Add, ac, r, ty)
    }

    /// The least and the most its values can be, each rounded outward.
    fn ends(&mut self, out: &mut Block, a: &A, ty: TypeId) -> (ValueId, ValueId) {
        let r = self.rad(out, a, ty);
        let ac = self.builtin(out, Builtin::Abs, vec![a.c], ty);
        let m = self.bin(out, BinOp::Add, ac, r, ty);
        let k = self.k(out, ty, (self.n as f64 + 3.0) * EPS);
        let g = self.bin(out, BinOp::Mul, m, k, ty);
        let t = self.k(out, ty, self.tiny());
        let g = self.bin(out, BinOp::Add, g, t, ty);
        let r = self.bin(out, BinOp::Add, r, g, ty);
        let lo = self.bin(out, BinOp::Sub, a.c, r, ty);
        let hi = self.bin(out, BinOp::Add, a.c, r, ty);
        (lo, hi)
    }

    /// `a` with `err` grown by `extra` (a magnitude) and by its rounding: `ulps` ulps of its
    /// magnitude (the CPU's or the GPU's count), and a subnormal; held at `BIG`.
    fn grown(
        &mut self,
        out: &mut Block,
        a: A,
        ty: TypeId,
        ulps: (f64, f64),
        extra: Option<ValueId>,
    ) -> A {
        let m = self.magnitude(out, &a, ty);
        let k = self.k(out, ty, self.ulps(ulps));
        let g = self.bin(out, BinOp::Mul, m, k, ty);
        let t = self.k(out, ty, self.tiny());
        let g = self.bin(out, BinOp::Add, g, t, ty);
        let mut e = self.bin(out, BinOp::Add, a.e, g, ty);
        if let Some(x) = extra {
            e = self.bin(out, BinOp::Add, e, x, ty);
        }
        let big = self.k(out, ty, BIG);
        let e = self.builtin(out, Builtin::Min, vec![e, big], ty);
        A { c: a.c, l: a.l, e }
    }

    fn rounded(&mut self, out: &mut Block, a: A, ty: TypeId, ulps: (f64, f64)) -> A {
        self.grown(out, a, ty, ulps, None)
    }

    /// The form of the range `[lo, hi]`, no slopes.
    fn of_ends(&mut self, out: &mut Block, lo: ValueId, hi: ValueId, ty: TypeId) -> A {
        let big = self.k(out, ty, BIG);
        let nbig = self.k(out, ty, -BIG);
        let lo = self.builtin(out, Builtin::Max, vec![lo, nbig], ty);
        let hi = self.builtin(out, Builtin::Min, vec![hi, big], ty);
        let half = self.k(out, ty, 0.5);
        let hl = self.bin(out, BinOp::Mul, lo, half, ty);
        let hh = self.bin(out, BinOp::Mul, hi, half, ty);
        let c = self.bin(out, BinOp::Add, hl, hh, ty);
        let e = self.bin(out, BinOp::Sub, hh, hl, ty);
        let z = self.zero(out, ty);
        self.rounded(out, A { c, l: vec![z; self.n], e }, ty, (2.0, 2.0))
    }

    fn add(&mut self, out: &mut Block, a: &A, b: &A, ty: TypeId, sub: bool) -> A {
        let op = if sub { BinOp::Sub } else { BinOp::Add };
        if self.is_exact(b) || (self.is_exact(a) && !sub) {
            // A shift: only the centre moves (and rounds, as the point code's sum does).
            let (x, k) = if self.is_exact(b) { (a, b.c) } else { (b, a.c) };
            let c = self.bin(out, op, x.c, k, ty);
            let r = A { c, l: x.l.clone(), e: x.e };
            return self.rounded(out, r, ty, (1.125, 1.125));
        }
        if self.is_exact(a) {
            // k - b: -b shifted.
            let nb = self.neg(out, b, ty);
            let c = self.bin(out, BinOp::Sub, a.c, b.c, ty);
            let r = A { c, l: nb.l, e: b.e };
            return self.rounded(out, r, ty, (1.125, 1.125));
        }
        let c = self.bin(out, op, a.c, b.c, ty);
        let l = (0..self.n).map(|i| self.bin(out, op, a.l[i], b.l[i], ty)).collect();
        let e = self.bin(out, BinOp::Add, a.e, b.e, ty);
        // Half an ulp each for the form's parts' sums, and half for the point code's sum, of the
        // result's magnitude (WGSL's sums are correctly rounded too), and an eighth for the
        // magnitude's own rounding.
        self.rounded(out, A { c, l, e }, ty, (1.125, 1.125))
    }

    fn neg(&mut self, out: &mut Block, a: &A, ty: TypeId) -> A {
        let c = self.emit(out, ty, Expr::Unary(UnOp::Neg, a.c));
        let l = (0..self.n).map(|i| self.emit(out, ty, Expr::Unary(UnOp::Neg, a.l[i]))).collect();
        A { c, l, e: a.e }
    }

    /// `a × k`, `k` inactive (of `ty`).
    fn scale(&mut self, out: &mut Block, a: &A, k: ValueId, ty: TypeId) -> A {
        let c = self.bin(out, BinOp::Mul, a.c, k, ty);
        let l = (0..self.n).map(|i| self.bin(out, BinOp::Mul, a.l[i], k, ty)).collect();
        let ak = self.builtin(out, Builtin::Abs, vec![k], ty);
        let e = self.bin(out, BinOp::Mul, a.e, ak, ty);
        // As a sum: half an ulp for the parts' products, half for the point code's.
        self.rounded(out, A { c, l, e }, ty, (1.125, 1.125))
    }

    /// `a ÷ k`, `k` inactive.
    fn divide(&mut self, out: &mut Block, a: &A, k: ValueId, ty: TypeId) -> A {
        let c = self.bin(out, BinOp::Div, a.c, k, ty);
        let l = (0..self.n).map(|i| self.bin(out, BinOp::Div, a.l[i], k, ty)).collect();
        let ak = self.builtin(out, Builtin::Abs, vec![k], ty);
        let e = self.bin(out, BinOp::Div, a.e, ak, ty);
        // WASM's division is correctly rounded; WGSL's within 2.5 ulp, the parts' and the
        // point code's each.
        self.rounded(out, A { c, l, e }, ty, (1.125, 5.25))
    }

    /// `a × b`: `a.c b.c + a.c b.l + b.c a.l`, and in `err` each error through the other's
    /// centre and the product of the radii; every number where either reaches past `CAP`.
    fn mul(&mut self, out: &mut Block, a: &A, b: &A, ty: TypeId) -> A {
        if self.is_exact(b) {
            return self.scale(out, a, b.c, ty);
        }
        if self.is_exact(a) {
            return self.scale(out, b, a.c, ty);
        }
        let (a, oka) = self.held(out, a, ty);
        let (b, okb) = self.held(out, b, ty);
        let (a, b) = (&a, &b);
        let c = self.bin(out, BinOp::Mul, a.c, b.c, ty);
        let mut l = Vec::new();
        for i in 0..self.n {
            let x = self.bin(out, BinOp::Mul, a.c, b.l[i], ty);
            let y = self.bin(out, BinOp::Mul, b.c, a.l[i], ty);
            l.push(self.bin(out, BinOp::Add, x, y, ty));
        }
        let ra = self.rad(out, a, ty);
        let rb = self.rad(out, b, ty);
        let cap = self.k(out, ty, CAP);
        let ra = self.builtin(out, Builtin::Min, vec![ra, cap], ty);
        let rb = self.builtin(out, Builtin::Min, vec![rb, cap], ty);
        let cross = self.bin(out, BinOp::Mul, ra, rb, ty);
        let aca = self.builtin(out, Builtin::Abs, vec![a.c], ty);
        let acb = self.builtin(out, Builtin::Abs, vec![b.c], ty);
        let ea = self.bin(out, BinOp::Mul, acb, a.e, ty);
        let eb = self.bin(out, BinOp::Mul, aca, b.e, ty);
        let e = self.bin(out, BinOp::Add, ea, eb, ty);
        let e = self.bin(out, BinOp::Add, e, cross, ty);
        // Rounding, in units of u = 2⁻²⁴, by the operands' own parts rather than the result's
        // (a slope's two products can cancel; their rounding doesn't): the centre's product 1,
        // each slope's two products and sum 2 of `|a.c| |b.l| + |b.c| |a.l|`, the error's three
        // products and two sums 3 of it, the radii's own sums (n + 1) of their product, and the
        // point code's product 1 of `(|a.c| + ra) (|b.c| + rb)`: in all at most
        // `2 |a.c b.c| + 4 (|a.c| rb + |b.c| ra) + (n + 5) ra rb`.
        let cc = self.bin(out, BinOp::Mul, aca, acb, ty);
        let x = self.bin(out, BinOp::Mul, aca, rb, ty);
        let y = self.bin(out, BinOp::Mul, acb, ra, ty);
        let xy = self.bin(out, BinOp::Add, x, y, ty);
        let k2 = self.k(out, ty, 2.0 * U);
        let k4 = self.k(out, ty, 4.0 * U);
        let kn = self.k(out, ty, (self.n as f64 + 5.0) * U);
        let m1 = self.bin(out, BinOp::Mul, cc, k2, ty);
        let m2 = self.bin(out, BinOp::Mul, xy, k4, ty);
        let m3 = self.bin(out, BinOp::Mul, cross, kn, ty);
        let m = self.bin(out, BinOp::Add, m1, m2, ty);
        let m = self.bin(out, BinOp::Add, m, m3, ty);
        // And the margin's own rounding: an ulp of it.
        let k = self.k(out, ty, 1.0 + 2.0 * EPS);
        let m = self.bin(out, BinOp::Mul, m, k, ty);
        let r = self.grown(out, A { c, l, e }, ty, (0.0, 0.0), Some(m));
        // An operand past `CAP` (every number) makes every number of the product: its radius
        // was held to `CAP` above only to keep the arithmetic finite, which bounds nothing.
        let bt = self.bool_ty;
        let ok = self.emit(out, bt, Expr::Binary(BinOp::And, oka, okb));
        let every = self.every(out, ty);
        self.pick(out, ok, &r, &every)
    }

    /// `a`, or every number where it reaches past `CAP` (so a product of two stays finite and
    /// makes no `∞ × 0`), and whether it reaches no further than `CAP` (`within`).
    fn held(&mut self, out: &mut Block, a: &A, ty: TypeId) -> (A, ValueId) {
        let ok = self.within(out, a);
        let every = self.every(out, ty);
        (self.pick(out, ok, a, &every), ok)
    }

    /// Whether every value of `a` is within `CAP` of 0 (a `bool`; false for a NaN). One that
    /// reaches past it is taken as every number, as an error of `BIG` is: `BIG` stands for no
    /// bound, so what a rule makes of it (a scaling by a small number, a product, a root) is
    /// no bound either, however small the number it comes to.
    fn within(&mut self, out: &mut Block, a: &A) -> ValueId {
        let ty = self.ty(a.c);
        let m = self.magnitude(out, a, ty);
        // A vector's largest component (a select takes one `bool`): the whole vector is every
        // number where any component is.
        let f = self.m.types.f32();
        let m = match self.width(ty) {
            Some(k) => {
                let mut x = self.emit(out, f, Expr::Extract(m, 0));
                for j in 1..k as u32 {
                    let y = self.emit(out, f, Expr::Extract(m, j));
                    x = self.builtin(out, Builtin::Max, vec![x, y], f);
                }
                x
            }
            None => m,
        };
        let cap = self.k(out, f, CAP);
        self.cmp(out, BinOp::Le, m, cap)
    }

    /// `r`, or every number unless each of `forms` is `within` `CAP`.
    fn unless_unbounded(&mut self, out: &mut Block, forms: &[A], r: A, ty: TypeId) -> A {
        let bt = self.bool_ty;
        let mut ok = None;
        for a in forms {
            let w = self.within(out, a);
            ok = Some(match ok {
                Some(o) => self.emit(out, bt, Expr::Binary(BinOp::And, o, w)),
                None => w,
            });
        }
        let Some(ok) = ok else { return r };
        let every = self.every(out, ty);
        self.pick(out, ok, &r, &every)
    }

    /// Every number: centre 0, error `BIG`.
    fn every(&mut self, out: &mut Block, ty: TypeId) -> A {
        let z = self.zero(out, ty);
        let big = self.k(out, ty, BIG);
        A { c: z, l: vec![z; self.n], e: big }
    }

    /// A function of `a` that is convex or concave on `a`'s range `[lo, hi]`, as the line
    /// `alpha x + zeta` and `delta` about it: `f - alpha x`'s extremes on the range are at its
    /// ends and at `xs` (clamped into the range), where `f'(xs) = alpha`; `zeta` and `delta` are
    /// their middle and half their spread, so any `alpha` gives a sound form. `f` gives the
    /// function's values; `fu`: its rounding in ulps, CPU and GPU.
    #[allow(clippy::too_many_arguments)]
    fn chebyshev(
        &mut self,
        out: &mut Block,
        a: &A,
        (lo, hi): (ValueId, ValueId),
        alpha: ValueId,
        xs: ValueId,
        f: &mut dyn FnMut(&mut Self, &mut Block, ValueId) -> ValueId,
        ty: TypeId,
        (cpu, gpu): (f64, f64),
    ) -> A {
        let xs = self.builtin(out, Builtin::Clamp, vec![xs, lo, hi], ty);
        let mut g = Vec::new();
        for x in [lo, hi, xs] {
            let fx = f(self, out, x);
            let ax = self.bin(out, BinOp::Mul, alpha, x, ty);
            g.push(self.bin(out, BinOp::Sub, fx, ax, ty));
        }
        let gmin = self.builtin(out, Builtin::Min, vec![g[0], g[1]], ty);
        let gmin = self.builtin(out, Builtin::Min, vec![gmin, g[2]], ty);
        let gmax = self.builtin(out, Builtin::Max, vec![g[0], g[1]], ty);
        let gmax = self.builtin(out, Builtin::Max, vec![gmax, g[2]], ty);
        let half = self.k(out, ty, 0.5);
        let zl = self.bin(out, BinOp::Mul, gmin, half, ty);
        let zh = self.bin(out, BinOp::Mul, gmax, half, ty);
        let zeta = self.bin(out, BinOp::Add, zl, zh, ty);
        let delta = self.bin(out, BinOp::Sub, zh, zl, ty);
        let ac = self.bin(out, BinOp::Mul, a.c, alpha, ty);
        let c = self.bin(out, BinOp::Add, ac, zeta, ty);
        let l = (0..self.n).map(|i| self.bin(out, BinOp::Mul, a.l[i], alpha, ty)).collect();
        let aa = self.builtin(out, Builtin::Abs, vec![alpha], ty);
        let e = self.bin(out, BinOp::Mul, a.e, aa, ty);
        let e = self.bin(out, BinOp::Add, e, delta, ty);
        // Where `|alpha x|` is far larger than the values (a steep line far from 0), its
        // rounding: ulps of it.
        let ax = self.builtin(out, Builtin::Abs, vec![lo], ty);
        let bx = self.builtin(out, Builtin::Abs, vec![hi], ty);
        let mx = self.builtin(out, Builtin::Max, vec![ax, bx], ty);
        let am = self.bin(out, BinOp::Mul, aa, mx, ty);
        let k = self.k(out, ty, self.ulps((cpu + 3.0, gpu + 3.0)));
        let am = self.bin(out, BinOp::Mul, am, k, ty);
        // Rounding, with `M` the result's magnitude and `kf` the function's own ulps: `f` at the
        // three points (2 kf ulps of `M` in all, the ends being values the form holds) and at the
        // point code's (kf more, counted with them), the line's products `alpha x` and sums
        // (`kf + 3` ulps of `|alpha| max(|lo|, |hi|)`, above), and its centre, slopes and error
        // (2 more ulps of `M`, and half for the margin's own).
        self.grown(out, A { c, l, e }, ty, (2.0 * cpu + 2.5, 2.0 * gpu + 2.5), Some(am))
    }

    /// The slope between `(lo, f(lo))` and `(hi, f(hi))` (`hi > lo` always: the ends are
    /// rounded apart).
    fn slope(
        &mut self,
        out: &mut Block,
        (lo, hi): (ValueId, ValueId),
        (fl, fh): (ValueId, ValueId),
        ty: TypeId,
    ) -> ValueId {
        let num = self.bin(out, BinOp::Sub, fh, fl, ty);
        let den = self.bin(out, BinOp::Sub, hi, lo, ty);
        let t = self.k(out, ty, 1.0e-37);
        let den = self.builtin(out, Builtin::Max, vec![den, t], ty);
        self.bin(out, BinOp::Div, num, den, ty)
    }

    fn sqrt(&mut self, out: &mut Block, a: &A, ty: TypeId) -> A {
        let (lo, hi) = self.ends(out, a, ty);
        let z = self.k(out, ty, 0.0);
        let lo = self.builtin(out, Builtin::Max, vec![lo, z], ty);
        let hi = self.builtin(out, Builtin::Max, vec![hi, lo], ty);
        let fl = self.builtin(out, Builtin::Sqrt, vec![lo], ty);
        let fh = self.builtin(out, Builtin::Sqrt, vec![hi], ty);
        let alpha = self.slope(out, (lo, hi), (fl, fh), ty);
        // Concave: f - alpha x is most where 1 / (2 sqrt x) = alpha, x = 1 / (4 alpha²).
        let a2 = self.bin(out, BinOp::Mul, alpha, alpha, ty);
        let four = self.k(out, ty, 4.0);
        let a24 = self.bin(out, BinOp::Mul, a2, four, ty);
        let t = self.k(out, ty, 1.0e-30);
        let a24 = self.builtin(out, Builtin::Max, vec![a24, t], ty);
        let one = self.k(out, ty, 1.0);
        let xs = self.bin(out, BinOp::Div, one, a24, ty);
        let mut f =
            |s: &mut Self, out: &mut Block, x: ValueId| s.builtin(out, Builtin::Sqrt, vec![x], ty);
        // WGSL: inherited from 1 / inverseSqrt (2 + 2.5 ulp); WASM: correctly rounded.
        self.chebyshev(out, a, (lo, hi), alpha, xs, &mut f, ty, (0.5, 5.0))
    }

    fn exp(&mut self, out: &mut Block, a: &A, ty: TypeId) -> A {
        if let Some(r) = self.per_comp(out, &[a], ty, |s, out, x, t| s.exp(out, &x[0], t)) {
            return r;
        }
        let (lo0, hi0) = self.ends(out, a, ty);
        // Held to [-100, 80]: past 80 it may overflow (every number); below -100 it is under
        // `exp(-100)`, which the line's drop from -100 to the range's end covers.
        let floor = self.k(out, ty, -100.0);
        let ceil = self.k(out, ty, 80.0);
        let lo = self.builtin(out, Builtin::Clamp, vec![lo0, floor, ceil], ty);
        let hi = self.builtin(out, Builtin::Clamp, vec![hi0, lo, ceil], ty);
        let fl = self.builtin(out, Builtin::Exp, vec![lo], ty);
        let fh = self.builtin(out, Builtin::Exp, vec![hi], ty);
        let alpha = self.slope(out, (lo, hi), (fl, fh), ty);
        // Convex: least where exp x = alpha.
        let t = self.k(out, ty, 1.0e-37);
        let al = self.builtin(out, Builtin::Max, vec![alpha, t], ty);
        let xs = self.builtin(out, Builtin::Log, vec![al], ty);
        let mut f =
            |s: &mut Self, out: &mut Block, x: ValueId| s.builtin(out, Builtin::Exp, vec![x], ty);
        // std's `exp` on the CPU: within 4 ulps (the interval's count); the GPU's 3 + 2|x|,
        // the 3 here and the rest below.
        let r = self.chebyshev(out, a, (lo, hi), alpha, xs, &mut f, ty, (4.0, 3.0));
        // Below -100 (where the range was held): `exp` there is between 0 and `exp(lo)`, and the
        // line falls by `alpha` times the distance held.
        let below = self.bin(out, BinOp::Sub, lo, lo0, ty);
        let drop = self.bin(out, BinOp::Mul, below, alpha, ty);
        let drop = self.bin(out, BinOp::Add, drop, fl, ty);
        let held = self.cmp(out, BinOp::Lt, lo0, floor);
        let z = self.zero(out, ty);
        let drop = self.sel(out, held, drop, z);
        let r = self.grown(out, r, ty, (0.0, 0.0), Some(drop));
        let fits = self.cmp(out, BinOp::Le, hi0, ceil);
        let every = self.every(out, ty);
        let r = self.pick(out, fits, &r, &every);
        if !self.gpu() {
            return r;
        }
        // WGSL: 3 + 2|x| ulp of the result, x held to ±200 (past which it's 0 or overflows).
        let al = self.builtin(out, Builtin::Abs, vec![lo], ty);
        let ah = self.builtin(out, Builtin::Abs, vec![hi], ty);
        let ax = self.builtin(out, Builtin::Max, vec![al, ah], ty);
        let lim = self.k(out, ty, 200.0);
        let ax = self.builtin(out, Builtin::Min, vec![ax, lim], ty);
        let four = self.k(out, ty, 4.0 * EPS);
        let six = self.k(out, ty, 6.0 * EPS);
        let kx = self.bin(out, BinOp::Mul, ax, four, ty);
        let kx = self.bin(out, BinOp::Add, kx, six, ty);
        let m = self.magnitude(out, &r, ty);
        let extra = self.bin(out, BinOp::Mul, m, kx, ty);
        self.grown(out, r, ty, (0.0, 0.0), Some(extra))
    }

    /// `1 / a`: on a range of one sign, convex (positive) or concave (negative), with
    /// `-1 / x² = alpha` at `xs`; a range that holds 0, every number.
    fn recip(&mut self, out: &mut Block, a: &A, ty: TypeId) -> A {
        if let Some(r) = self.per_comp(out, &[a], ty, |s, out, x, t| s.recip(out, &x[0], t)) {
            return r;
        }
        let (lo, hi) = self.ends(out, a, ty);
        // Of one sign, and no nearer 0 than 2⁻¹⁰⁰ (so 1 / x stays finite); else every number,
        // and the line is drawn over [1, 2] instead, to make no infinity.
        let near = self.k(out, ty, 7.9e-31);
        let nnear = self.k(out, ty, -7.9e-31);
        let pos = self.cmp(out, BinOp::Gt, lo, near);
        let neg = self.cmp(out, BinOp::Lt, hi, nnear);
        let bt = self.bool_ty;
        let ok = self.emit(out, bt, Expr::Binary(BinOp::Or, pos, neg));
        let one = self.k(out, ty, 1.0);
        let two = self.k(out, ty, 2.0);
        let lo = self.sel(out, ok, lo, one);
        let hi = self.sel(out, ok, hi, two);
        let fl = self.bin(out, BinOp::Div, one, lo, ty);
        let fh = self.bin(out, BinOp::Div, one, hi, ty);
        let alpha = self.slope(out, (lo, hi), (fl, fh), ty);
        let na = self.emit(out, ty, Expr::Unary(UnOp::Neg, alpha));
        let t = self.k(out, ty, 1.0e-30);
        let na = self.builtin(out, Builtin::Max, vec![na, t], ty);
        let inv = self.bin(out, BinOp::Div, one, na, ty);
        let s = self.builtin(out, Builtin::Sqrt, vec![inv], ty);
        let ns = self.emit(out, ty, Expr::Unary(UnOp::Neg, s));
        let xs = self.sel(out, neg, ns, s);
        let mut f = |s: &mut Self, out: &mut Block, x: ValueId| {
            let one = s.k(out, ty, 1.0);
            s.bin(out, BinOp::Div, one, x, ty)
        };
        let r = self.chebyshev(out, a, (lo, hi), alpha, xs, &mut f, ty, (0.5, 2.5));
        let every = self.every(out, ty);
        self.pick(out, ok, &r, &every)
    }

    /// The line under `|x|` over `[lo, hi]`, and how far above it `|x|` goes there:
    /// `|x| ∈ αx + [0, γ]`. Any `α` in [-1, 1] makes `|x| - αx` convex and never below 0, so
    /// its most is at an end: `γ = max(|lo| - α lo, |hi| - α hi)`, and the secant's slope, held
    /// to [-1, 1], makes it least (0 where the range has one sign). `γ` is widened by its own
    /// rounding, within 3u of `max(|lo|, |hi|)`. Componentwise; no choice, so no select.
    fn abs_line(
        &mut self,
        out: &mut Block,
        (lo, hi): (ValueId, ValueId),
        ty: TypeId,
    ) -> (ValueId, ValueId) {
        let s = self.bin(out, BinOp::Add, lo, hi, ty);
        let d = self.bin(out, BinOp::Sub, hi, lo, ty);
        let t = self.k(out, ty, 1.0e-37);
        let d = self.builtin(out, Builtin::Max, vec![d, t], ty);
        let q = self.bin(out, BinOp::Div, s, d, ty);
        let one = self.k(out, ty, 1.0);
        let none = self.k(out, ty, -1.0);
        let alpha = self.builtin(out, Builtin::Clamp, vec![q, none, one], ty);
        let al = self.builtin(out, Builtin::Abs, vec![lo], ty);
        let ah = self.builtin(out, Builtin::Abs, vec![hi], ty);
        let xl = self.bin(out, BinOp::Mul, alpha, lo, ty);
        let xh = self.bin(out, BinOp::Mul, alpha, hi, ty);
        let gl = self.bin(out, BinOp::Sub, al, xl, ty);
        let gh = self.bin(out, BinOp::Sub, ah, xh, ty);
        let g = self.builtin(out, Builtin::Max, vec![gl, gh], ty);
        let mx = self.builtin(out, Builtin::Max, vec![al, ah], ty);
        let k = self.k(out, ty, 3.0 * U);
        let r = self.bin(out, BinOp::Mul, mx, k, ty);
        let g = self.bin(out, BinOp::Add, g, r, ty);
        (alpha, g)
    }

    /// `|a|`: `α a + γ/2 ± γ/2` (`abs_line`): itself or its negation, exactly, where its range has
    /// one sign. The point code's `abs` is exact; the form's own arithmetic rounds within an ulp
    /// of `|a.c| + rad + γ`.
    fn abs(&mut self, out: &mut Block, a: &A, ty: TypeId) -> A {
        let (lo, hi) = self.ends(out, a, ty);
        let (alpha, g) = self.abs_line(out, (lo, hi), ty);
        let half = self.k(out, ty, 0.5);
        let g2 = self.bin(out, BinOp::Mul, g, half, ty);
        let ac = self.bin(out, BinOp::Mul, alpha, a.c, ty);
        let c = self.bin(out, BinOp::Add, ac, g2, ty);
        let l = (0..self.n).map(|i| self.bin(out, BinOp::Mul, alpha, a.l[i], ty)).collect();
        let aa = self.builtin(out, Builtin::Abs, vec![alpha], ty);
        let e = self.bin(out, BinOp::Mul, aa, a.e, ty);
        let e = self.bin(out, BinOp::Add, e, g2, ty);
        let m = self.magnitude(out, a, ty);
        let m = self.bin(out, BinOp::Add, m, g, ty);
        let k = self.k(out, ty, EPS);
        let m = self.bin(out, BinOp::Mul, m, k, ty);
        self.grown(out, A { c, l, e }, ty, (0.0, 0.0), Some(m))
    }

    /// `t` where `cond`, else `f`, part by part.
    fn pick(&mut self, out: &mut Block, cond: ValueId, t: &A, f: &A) -> A {
        let c = self.sel(out, cond, t.c, f.c);
        let l = (0..self.n).map(|i| self.sel(out, cond, t.l[i], f.l[i])).collect();
        let e = self.sel(out, cond, t.e, f.e);
        A { c, l, e }
    }

    /// A form holding both `a`'s values and `b`'s: their mean slopes (or the other's, where one
    /// is empty), and about them the union of what each strays from those.
    fn hull(&mut self, out: &mut Block, a: &A, b: &A, ty: TypeId) -> A {
        if let Some(r) =
            self.per_comp(out, &[a, b], ty, |s, out, x, t| s.hull(out, &x[0], &x[1], t))
        {
            return r;
        }
        let z = self.k(out, ty, 0.0);
        let a_empty = self.cmp(out, BinOp::Lt, a.e, z);
        let b_empty = self.cmp(out, BinOp::Lt, b.e, z);
        let half = self.k(out, ty, 0.5);
        let mut l = Vec::new();
        for i in 0..self.n {
            let s = self.bin(out, BinOp::Add, a.l[i], b.l[i], ty);
            let mean = self.bin(out, BinOp::Mul, s, half, ty);
            let x = self.sel(out, b_empty, a.l[i], mean);
            l.push(self.sel(out, a_empty, b.l[i], x));
        }
        let span = |s: &mut Self, out: &mut Block, x: &A| -> (ValueId, ValueId) {
            let mut r = x.e;
            for (i, li) in l.iter().enumerate() {
                let d = s.bin(out, BinOp::Sub, x.l[i], *li, ty);
                let d = s.builtin(out, Builtin::Abs, vec![d], ty);
                r = s.bin(out, BinOp::Add, r, d, ty);
            }
            (s.bin(out, BinOp::Sub, x.c, r, ty), s.bin(out, BinOp::Add, x.c, r, ty))
        };
        let (al, ah) = span(self, out, a);
        let (bl, bh) = span(self, out, b);
        let lo = self.builtin(out, Builtin::Min, vec![al, bl], ty);
        let hi = self.builtin(out, Builtin::Max, vec![ah, bh], ty);
        let hl = self.bin(out, BinOp::Mul, lo, half, ty);
        let hh = self.bin(out, BinOp::Mul, hi, half, ty);
        let c = self.bin(out, BinOp::Add, hl, hh, ty);
        let e = self.bin(out, BinOp::Sub, hh, hl, ty);
        // Each span's sums (n + 2 roundings, of `M`'s size), the mean slopes', the ends'
        // halves and the centre's and error's: within (n + 6) u of the magnitude.
        let k = 0.5 * self.n as f64 + 3.0;
        let r = self.rounded(out, A { c, l, e }, ty, (k, k));
        // Both empty: still empty.
        let bt = self.bool_of(ty);
        let both = self.emit(out, bt, Expr::Binary(BinOp::And, a_empty, b_empty));
        let none = self.empty(out, ty);
        self.pick(out, both, &none, &r)
    }

    /// `t` where the condition is surely true, `f` where surely false, else their hull.
    fn join(&mut self, out: &mut Block, (st, sf): Sure, t: &A, f: &A, ty: TypeId) -> A {
        let h = self.hull(out, t, f, ty);
        let r = self.pick(out, sf, f, &h);
        self.pick(out, st, t, &r)
    }

    fn join_bool(&mut self, out: &mut Block, (st, sf): Sure, t: Sure, f: Sure) -> Sure {
        let bt = self.bool_ty;
        let both_t = self.emit(out, bt, Expr::Binary(BinOp::And, t.0, f.0));
        let both_f = self.emit(out, bt, Expr::Binary(BinOp::And, t.1, f.1));
        let x = self.sel(out, sf, f.0, both_t);
        let y = self.sel(out, sf, f.1, both_f);
        (self.sel(out, st, t.0, x), self.sel(out, st, t.1, y))
    }

    /// `min(a, b)` or `max`: `(a + b ∓ |a - b|) / 2` with `|a - b|` as `abs_line` gives it,
    /// so the weights `(1 ∓ α) / 2` and `(1 ± α) / 2` of `a` and `b`, and `γ / 4` about them:
    /// `a` or `b` itself where a box surely orders them (`α = ±1`, `γ` only its rounding). The
    /// point code's is exact; the form's arithmetic rounds within 4u of `|a| + |b|` (the
    /// weights' own rounding included) and 2u of `γ`.
    fn min_max(&mut self, out: &mut Block, a: &A, b: &A, ty: TypeId, max: bool) -> A {
        let dc = self.bin(out, BinOp::Sub, a.c, b.c, ty);
        let dl = (0..self.n).map(|i| self.bin(out, BinOp::Sub, a.l[i], b.l[i], ty)).collect();
        let de = self.bin(out, BinOp::Add, a.e, b.e, ty);
        let d = A { c: dc, l: dl, e: de };
        let (lo, hi) = self.ends(out, &d, ty);
        let (alpha, g) = self.abs_line(out, (lo, hi), ty);
        let half = self.k(out, ty, 0.5);
        let ha = self.bin(out, BinOp::Mul, alpha, half, ty);
        let lower = self.bin(out, BinOp::Sub, half, ha, ty);
        let upper = self.bin(out, BinOp::Add, half, ha, ty);
        let (wa, wb) = if max { (upper, lower) } else { (lower, upper) };
        let quarter = self.k(out, ty, 0.25);
        let g4 = self.bin(out, BinOp::Mul, g, quarter, ty);
        let mix = |s: &mut Self, out: &mut Block, x: ValueId, y: ValueId| {
            let p = s.bin(out, BinOp::Mul, wa, x, ty);
            let q = s.bin(out, BinOp::Mul, wb, y, ty);
            s.bin(out, BinOp::Add, p, q, ty)
        };
        let c = mix(self, out, a.c, b.c);
        let c = self.bin(out, if max { BinOp::Add } else { BinOp::Sub }, c, g4, ty);
        let l = (0..self.n).map(|i| mix(self, out, a.l[i], b.l[i])).collect();
        let e = mix(self, out, a.e, b.e);
        let e = self.bin(out, BinOp::Add, e, g4, ty);
        let ma = self.magnitude(out, a, ty);
        let mb = self.magnitude(out, b, ty);
        let m = self.bin(out, BinOp::Add, ma, mb, ty);
        let k = self.k(out, ty, 2.0 * EPS);
        let m = self.bin(out, BinOp::Mul, m, k, ty);
        let kg = self.k(out, ty, EPS);
        let mg = self.bin(out, BinOp::Mul, g, kg, ty);
        let m = self.bin(out, BinOp::Add, m, mg, ty);
        self.grown(out, A { c, l, e }, ty, (0.0, 0.0), Some(m))
    }

    /// A comparison's certainty, from the range of `a - b` (a value against itself is sure).
    fn compare(&mut self, out: &mut Block, op: BinOp, (x, y): (ValueId, ValueId)) -> R<Sure> {
        if x == y {
            let t = matches!(op, BinOp::Eq | BinOp::Le | BinOp::Ge);
            return Ok((self.bconst(out, t), self.bconst(out, !t)));
        }
        let ty = self.meet(x, y);
        if !matches!(self.m.types.get(ty), TypeDef::Scalar(_)) {
            return Err(Error::not_derivable("a vector comparison of what depends on the input"));
        }
        let a = self.of_as(out, x, ty);
        let b = self.of_as(out, y, ty);
        let d = self.add(out, &a, &b, ty, true);
        let (lo, hi) = self.ends(out, &d, ty);
        let z = self.k(out, ty, 0.0);
        let (st, sf) = match op {
            BinOp::Lt => (self.cmp(out, BinOp::Lt, hi, z), self.cmp(out, BinOp::Ge, lo, z)),
            BinOp::Le => (self.cmp(out, BinOp::Le, hi, z), self.cmp(out, BinOp::Gt, lo, z)),
            BinOp::Gt => (self.cmp(out, BinOp::Gt, lo, z), self.cmp(out, BinOp::Le, hi, z)),
            BinOp::Ge => (self.cmp(out, BinOp::Ge, lo, z), self.cmp(out, BinOp::Lt, hi, z)),
            BinOp::Eq | BinOp::Ne => {
                let above = self.cmp(out, BinOp::Gt, lo, z);
                let below = self.cmp(out, BinOp::Lt, hi, z);
                let bt = self.bool_ty;
                let apart = self.emit(out, bt, Expr::Binary(BinOp::Or, above, below));
                let never = self.bconst(out, false);
                if op == BinOp::Eq { (never, apart) } else { (apart, never) }
            }
            _ => return Err(Error::not_derivable("a comparison with no affine rule")),
        };
        // An operand of no bound (`within`) decides nothing.
        let active: Vec<A> = [x, y].iter().filter_map(|v| self.forms.get(v).cloned()).collect();
        let mut sure = (st, sf);
        for a in &active {
            let ok = self.within(out, a);
            let bt = self.bool_ty;
            sure.0 = self.emit(out, bt, Expr::Binary(BinOp::And, sure.0, ok));
            sure.1 = self.emit(out, bt, Expr::Binary(BinOp::And, sure.1, ok));
        }
        Ok(sure)
    }

    /// The type two operands meet at (the vector, where one is a scalar).
    fn meet(&self, a: ValueId, b: ValueId) -> TypeId {
        let (ta, tb) = (self.f.value_ty(a), self.f.value_ty(b));
        match self.m.types.get(ta) {
            TypeDef::Vector(..) => ta,
            _ => tb,
        }
    }

    // ---- statements ------------------------------------------------------------------------

    fn block(&mut self, b: &Block) -> R<Block> {
        let mut out = Vec::new();
        self.block_into(b, &mut out)?;
        Ok(out)
    }

    fn block_into(&mut self, b: &Block, out: &mut Block) -> R<()> {
        for s in b {
            self.stmt(s, out)?;
        }
        Ok(())
    }

    fn active_root(&self, r: &PlaceRoot) -> bool {
        match r {
            PlaceRoot::Local(l) => !self.lparts[l.index()].is_empty(),
            PlaceRoot::Param(i) => !self.pparts[*i as usize].is_empty(),
            _ => false,
        }
    }

    /// The places that hold an active place's parts: its slopes' and error's, or a bool's
    /// `sure` and `never`.
    fn place_parts(&self, p: &Place) -> R<Vec<Place>> {
        let roots: Vec<PlaceRoot> = match &p.root {
            PlaceRoot::Local(l) => {
                self.lparts[l.index()].iter().map(|x| PlaceRoot::Local(*x)).collect()
            }
            PlaceRoot::Param(i) => {
                self.pparts[*i as usize].iter().map(|x| PlaceRoot::Param(*x)).collect()
            }
            _ => return Err(Error::not_derivable("an affine form can't follow this place")),
        };
        if roots.is_empty() {
            return Err(Error::internal("an inactive place read as active"));
        }
        Ok(roots.into_iter().map(|root| Place { root, path: p.path.clone() }).collect())
    }

    fn is_bool_root(&self, r: &PlaceRoot) -> bool {
        match r {
            PlaceRoot::Local(l) => self.f.locals[l.index()].ty == self.bool_ty,
            _ => false,
        }
    }

    fn load_form(&mut self, out: &mut Block, p: &Place, ty: TypeId) -> R<A> {
        let c = self.emit(out, ty, Expr::Load(p.clone()));
        let mut ps = Vec::new();
        for q in self.place_parts(p)? {
            ps.push(self.emit(out, ty, Expr::Load(q)));
        }
        let e = ps.pop().ok_or_else(|| Error::internal("no error part"))?;
        Ok(A { c, l: ps, e })
    }

    fn store_form(&mut self, out: &mut Block, p: &Place, a: &A) -> R<()> {
        out.push(Stmt::Store(p.clone(), a.c));
        for (k, q) in self.place_parts(p)?.into_iter().enumerate() {
            out.push(Stmt::Store(q, if k < self.n { a.l[k] } else { a.e }));
        }
        Ok(())
    }

    fn load_sure(&mut self, out: &mut Block, p: &Place) -> R<Sure> {
        let bt = self.bool_ty;
        let ps = self.place_parts(p)?;
        let t = self.emit(out, bt, Expr::Load(ps[0].clone()));
        let f = self.emit(out, bt, Expr::Load(ps[1].clone()));
        Ok((t, f))
    }

    fn store_sure(&mut self, out: &mut Block, p: &Place, (t, f): Sure) -> R<()> {
        let ps = self.place_parts(p)?;
        out.push(Stmt::Store(ps[0].clone(), t));
        out.push(Stmt::Store(ps[1].clone(), f));
        Ok(())
    }

    /// The locals and parameters a block may write: by a store, or lent to a callee's `mut`
    /// parameter. (A write through a pointer into an active place is refused where it is.)
    fn written(&self, b: &Block, into: &mut Vec<PlaceRoot>) {
        let mut add = |r: &PlaceRoot| {
            if matches!(r, PlaceRoot::Local(_) | PlaceRoot::Param(_)) && !into.contains(r) {
                into.push(r.clone());
            }
        };
        crate::visit::walk(b, &mut |s| match s {
            Stmt::Store(p, _) => add(&p.root),
            Stmt::Let(_, Expr::Call(g, args)) | Stmt::Eval(Expr::Call(g, args)) => {
                for (a, p) in args.iter().zip(&self.m.functions[g.index()].params) {
                    if let (Arg::Place(pl), true) = (a, p.mutable) {
                        add(&pl.root);
                    }
                }
            }
            _ => {}
        });
    }

    /// The type of a local or a parameter.
    fn root_ty(&self, r: &PlaceRoot) -> R<TypeId> {
        match r {
            PlaceRoot::Local(l) => Ok(self.f.locals[l.index()].ty),
            PlaceRoot::Param(i) => Ok(self.f.params[*i as usize].ty),
            _ => Err(Error::internal("a joined place that isn't a local or a parameter")),
        }
    }

    /// Whether a write to a place with root `r` may reach an active local or parameter through
    /// a pointer (any, where it isn't known where it points).
    fn reaches_active(&self, r: &PlaceRoot) -> bool {
        let PlaceRoot::Ptr(v) = r else { return false };
        let mut roots = Vec::new();
        if self.ptrs.roots(*v, &mut roots).is_none() {
            return self.lparts.iter().any(|p| !p.is_empty())
                || self.pparts.iter().any(|p| !p.is_empty());
        }
        roots.iter().any(|r| self.active_root(r))
    }

    /// Whether a call lends a `mut` parameter a place through a pointer that may reach an
    /// active local or parameter.
    fn lends_through_pointer(&self, g: FuncId, args: &[Arg]) -> bool {
        args.iter()
            .zip(&self.m.functions[g.index()].params)
            .any(|(a, p)| matches!(a, Arg::Place(pl) if p.mutable && self.reaches_active(&pl.root)))
    }

    /// A local's or parameter's whole state: its form, or its certainty.
    fn load_state(&mut self, out: &mut Block, r: &PlaceRoot) -> R<State> {
        let p = Place::root(r.clone());
        if self.is_bool_root(r) {
            return Ok(State::Bool(self.load_sure(out, &p)?));
        }
        let ty = self.root_ty(r)?;
        Ok(State::Float(self.load_form(out, &p, ty)?))
    }

    fn store_state(&mut self, out: &mut Block, r: &PlaceRoot, s: &State) -> R<()> {
        let p = Place::root(r.clone());
        match s {
            State::Bool(b) => self.store_sure(out, &p, *b),
            State::Float(a) => self.store_form(out, &p, a),
        }
    }

    fn stmt(&mut self, s: &Stmt, out: &mut Block) -> R<()> {
        match s {
            Stmt::Let(_, Expr::Call(g, args)) | Stmt::Eval(Expr::Call(g, args))
                if self.lends_through_pointer(*g, args) =>
            {
                Err(Error::not_derivable(
                    "a call that may write through a pointer into what depends on the input",
                ))
            }
            Stmt::Let(v, Expr::Call(g, args)) if self.any_active(args) => {
                self.call(Some(*v), *g, args, out)
            }
            Stmt::Eval(Expr::Call(g, args)) if self.any_active(args) => {
                self.call(None, *g, args, out)
            }
            Stmt::Let(v, e) if self.act.value(*v) => self.let_(*v, e, out),
            Stmt::Store(p, v) if self.active_root(&p.root) => {
                if self.is_bool_root(&p.root) {
                    let s = self.sure_of(out, *v);
                    return self.store_sure(out, p, s);
                }
                let ty = place_type(self.m, self.f, p)?;
                let a = self.of_as(out, *v, ty);
                self.store_form(out, p, &a)
            }
            Stmt::Store(_, v) if self.act.value(*v) => {
                Err(Error::not_derivable("a store of what depends on the input into another place"))
            }
            Stmt::Store(p, _) if self.reaches_active(&p.root) => Err(Error::not_derivable(
                "a store through a pointer that may reach what depends on the input",
            )),
            Stmt::If { cond, then, else_ } if self.act.value(*cond) => {
                let sure = self.sure_of(out, *cond);
                let mut roots = Vec::new();
                self.written(then, &mut roots);
                self.written(else_, &mut roots);
                // A join is of floats (or bools): an aggregate written under the branch isn't.
                for r in &roots {
                    let t = self.root_ty(r)?;
                    if !self.active_root(r) || (t != self.bool_ty && !is_float(&self.m.types, t)) {
                        return Err(Error::not_derivable(
                            "a branch on the input that writes a place that isn't a float or a \
                             bool",
                        ));
                    }
                }
                let mut before = Vec::new();
                for r in &roots {
                    before.push(self.load_state(out, r)?);
                }
                self.speculative += 1;
                self.block_into(then, out)?;
                let mut after = Vec::new();
                for (r, b) in roots.iter().zip(&before) {
                    after.push(self.load_state(out, r)?);
                    self.store_state(out, r, b)?;
                }
                self.block_into(else_, out)?;
                self.speculative -= 1;
                for (r, t) in roots.iter().zip(after) {
                    let e = self.load_state(out, r)?;
                    let j = match (t, e) {
                        (State::Float(t), State::Float(e)) => {
                            let ty = self.root_ty(r)?;
                            State::Float(self.join(out, sure, &t, &e, ty))
                        }
                        (State::Bool(t), State::Bool(e)) => {
                            State::Bool(self.join_bool(out, sure, t, e))
                        }
                        _ => return Err(Error::internal("a local's state changed kind")),
                    };
                    self.store_state(out, r, &j)?;
                }
                Ok(())
            }
            Stmt::Return(Some(v)) if self.returns => {
                let ty = self.f.ret.ok_or_else(|| Error::internal("no result type"))?;
                let a = self.of_as(out, *v, ty);
                let rt = self.nf.ret.ok_or_else(|| Error::internal("no affine type"))?;
                let mut fields = vec![a.c];
                fields.extend(a.l.iter().copied());
                fields.push(a.e);
                let r = self.emit(out, rt, Expr::Construct(rt, fields));
                out.push(Stmt::Return(Some(r)));
                Ok(())
            }
            Stmt::Trap if self.speculative > 0 => Ok(()),
            other => pass_through(other, out, &mut |b| self.block(b)),
        }
    }

    fn any_active(&self, args: &[Arg]) -> bool {
        args.iter().any(|a| arg_bits(self.m, self.f, &self.act, a) != 0)
    }

    fn let_(&mut self, v: ValueId, e: &Expr, out: &mut Block) -> R<()> {
        let ty = self.f.value_ty(v);
        if ty == self.bool_ty {
            let s = self.let_bool(e, out)?;
            self.sure.insert(v, s);
            return Ok(());
        }
        if !floaty(&self.m.types, ty) {
            return Err(Error::not_derivable(
                "an active value that isn't made of floats or a bool",
            ));
        }
        let aggregate = !is_float(&self.m.types, ty);
        if aggregate
            && !matches!(
                e,
                Expr::Param(_) | Expr::Load(_) | Expr::Extract(..) | Expr::Construct(..)
            )
            && !matches!(e, Expr::Select { cond, .. } if !self.act.value(*cond))
            && !matches!(e, Expr::Call(..))
        {
            return Err(Error::not_derivable(
                "arithmetic on an aggregate that depends on the input",
            ));
        }
        let a = match e {
            Expr::Param(i) if !self.pparts[*i as usize].is_empty() => {
                let c = self.emit(out, ty, Expr::Param(*i));
                let ps = self.pparts[*i as usize].clone();
                let mut vs: Vec<ValueId> =
                    ps.iter().map(|k| self.emit(out, ty, Expr::Param(*k))).collect();
                let e = vs.pop().ok_or_else(|| Error::internal("no error part"))?;
                A { c, l: vs, e }
            }
            Expr::Load(p) if self.active_root(&p.root) => self.load_form(out, p, ty)?,
            Expr::Unary(UnOp::Neg, x) => {
                let a = self.of_as(out, *x, ty);
                self.neg(out, &a, ty)
            }
            Expr::Binary(op @ (BinOp::Add | BinOp::Sub), a, b) => {
                let (aa, bb) = (self.of_as(out, *a, ty), self.of_as(out, *b, ty));
                self.add(out, &aa, &bb, ty, *op == BinOp::Sub)
            }
            Expr::Binary(BinOp::Mul, a, b) => self.product(out, *a, *b, ty)?,
            Expr::Binary(BinOp::Div, a, b) => {
                let aa = self.of_as(out, *a, ty);
                if !self.act.value(*b) {
                    let k = self.coerce(out, *b, ty);
                    self.divide(out, &aa, k, ty)
                } else {
                    let bb = self.of_as(out, *b, ty);
                    let r = self.recip(out, &bb, ty);
                    let q = self.mul(out, &aa, &r, ty);
                    // The point code divides once: its rounding, which `a × (1 / b)` isn't.
                    self.rounded(out, q, ty, (1.0, 3.0))
                }
            }
            Expr::Select { cond, if_true, if_false } => {
                let (t, f) = (self.of_as(out, *if_true, ty), self.of_as(out, *if_false, ty));
                if self.act.value(*cond) {
                    let sure = self.sure_of(out, *cond);
                    self.join(out, sure, &t, &f, ty)
                } else {
                    self.pick(out, *cond, &t, &f)
                }
            }
            Expr::Extract(x, k) => {
                let a = self.of(out, *x);
                let c = self.emit(out, ty, Expr::Extract(a.c, *k));
                let l =
                    (0..self.n).map(|i| self.emit(out, ty, Expr::Extract(a.l[i], *k))).collect();
                let e = self.emit(out, ty, Expr::Extract(a.e, *k));
                A { c, l, e }
            }
            Expr::Swizzle(x, sw) => {
                let a = self.of(out, *x);
                let c = self.emit(out, ty, Expr::Swizzle(a.c, sw.clone()));
                let l = (0..self.n)
                    .map(|i| self.emit(out, ty, Expr::Swizzle(a.l[i], sw.clone())))
                    .collect();
                let e = self.emit(out, ty, Expr::Swizzle(a.e, sw.clone()));
                A { c, l, e }
            }
            Expr::Construct(t, comps) if floaty(&self.m.types, *t) => {
                let cs: Vec<A> = comps.iter().map(|x| self.of(out, *x)).collect();
                let c = self.emit(out, ty, Expr::Construct(*t, cs.iter().map(|a| a.c).collect()));
                let l = (0..self.n)
                    .map(|i| {
                        self.emit(out, ty, Expr::Construct(*t, cs.iter().map(|a| a.l[i]).collect()))
                    })
                    .collect();
                let e = self.emit(out, ty, Expr::Construct(*t, cs.iter().map(|a| a.e).collect()));
                A { c, l, e }
            }
            Expr::Splat(x, k) => {
                let a = self.of(out, *x);
                let c = self.emit(out, ty, Expr::Splat(a.c, *k));
                let l = (0..self.n).map(|i| self.emit(out, ty, Expr::Splat(a.l[i], *k))).collect();
                let e = self.emit(out, ty, Expr::Splat(a.e, *k));
                A { c, l, e }
            }
            Expr::Builtin(b, args) => match self.builtin_rule(out, *b, args, ty)? {
                Some(a) => a,
                None => self.by_interval(out, e, ty)?,
            },
            Expr::Binary(..) | Expr::Unary(..) => self.by_interval(out, e, ty)?,
            _ => return Err(Error::not_derivable("an expression with no affine rule")),
        };
        // What can make a smaller number of no bound (`within`): a product, a quotient, a
        // builtin, an interval's rule. A sum, a negation and what only moves parts keep an
        // error of `BIG` as it is.
        let operands: Vec<ValueId> = match e {
            Expr::Binary(BinOp::Add | BinOp::Sub, ..) | Expr::Unary(UnOp::Neg, _) => Vec::new(),
            Expr::Binary(_, x, y) => vec![*x, *y],
            Expr::Unary(_, x) => vec![*x],
            Expr::Builtin(_, xs) => xs.clone(),
            _ => Vec::new(),
        };
        let forms: Vec<A> = operands
            .iter()
            .filter(|x| is_float(&self.m.types, self.f.value_ty(**x)))
            .filter_map(|x| self.forms.get(x).cloned())
            .collect();
        let a = self.unless_unbounded(out, &forms, a, ty);
        self.forms.insert(v, a);
        Ok(())
    }

    fn let_bool(&mut self, e: &Expr, out: &mut Block) -> R<Sure> {
        Ok(match e {
            Expr::Binary(
                op @ (BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne),
                a,
                b,
            ) if is_float(&self.m.types, self.f.value_ty(*a)) => {
                self.compare(out, *op, (*a, *b))?
            }
            Expr::Binary(op @ (BinOp::And | BinOp::Or), a, b) => {
                let (x, y) = (self.sure_of(out, *a), self.sure_of(out, *b));
                let bt = self.bool_ty;
                let (p, q) = if *op == BinOp::And {
                    (BinOp::And, BinOp::Or)
                } else {
                    (BinOp::Or, BinOp::And)
                };
                (
                    self.emit(out, bt, Expr::Binary(p, x.0, y.0)),
                    self.emit(out, bt, Expr::Binary(q, x.1, y.1)),
                )
            }
            Expr::Unary(UnOp::Not, x) => {
                let (t, f) = self.sure_of(out, *x);
                (f, t)
            }
            Expr::Load(p) if self.active_root(&p.root) && self.is_bool_root(&p.root) => {
                self.load_sure(out, p)?
            }
            Expr::Select { cond, if_true, if_false } => {
                let (t, f) = (self.sure_of(out, *if_true), self.sure_of(out, *if_false));
                let c = self.sure_of(out, *cond);
                self.join_bool(out, c, t, f)
            }
            _ => return Err(Error::not_derivable("a condition with no affine rule")),
        })
    }

    /// `a × b`: a scaling where one is inactive, a matrix's product with a vector, or a product
    /// of forms.
    fn product(&mut self, out: &mut Block, a: ValueId, b: ValueId, ty: TypeId) -> R<A> {
        let (ta, tb) = (self.f.value_ty(a), self.f.value_ty(b));
        let mat = |t: TypeId| matches!(self.m.types.get(t), TypeDef::Matrix(..));
        if mat(ta) || mat(tb) {
            let (m, v, left) = if mat(ta) { (a, b, true) } else { (b, a, false) };
            if self.act.value(m) {
                return Err(Error::not_derivable("a matrix that depends on the input"));
            }
            return Ok(self.mat_product(out, m, v, left, ty));
        }
        Ok(match (self.act.value(a), self.act.value(b)) {
            (true, false) => {
                let aa = self.of_as(out, a, ty);
                let k = self.coerce(out, b, ty);
                self.scale(out, &aa, k, ty)
            }
            (false, true) => {
                let bb = self.of_as(out, b, ty);
                let k = self.coerce(out, a, ty);
                self.scale(out, &bb, k, ty)
            }
            _ => {
                let (aa, bb) = (self.of_as(out, a, ty), self.of_as(out, b, ty));
                self.mul(out, &aa, &bb, ty)
            }
        })
    }

    /// `M v` (`left`) or `v M`, `M` inactive: exact on the slopes, `|M|` on the error, and the
    /// sums' rounding relative to `|M| (|c| + rad)`.
    fn mat_product(
        &mut self,
        out: &mut Block,
        mv: ValueId,
        v: ValueId,
        left: bool,
        ty: TypeId,
    ) -> A {
        let mt = self.f.value_ty(mv);
        let (cols, rows) = match self.m.types.get(mt) {
            &TypeDef::Matrix(c, r) => (c, r),
            _ => unreachable!("a matrix"),
        };
        let col_ty = self.m.types.intern(TypeDef::Vector(Scalar::F32, rows));
        let a = self.of(out, v);
        let vt = self.ty(a.c);
        let mut abs_cols = Vec::new();
        for j in 0..cols {
            let col = self.emit(out, col_ty, Expr::Extract(mv, j as u32));
            abs_cols.push(self.builtin(out, Builtin::Abs, vec![col], col_ty));
        }
        let am = self.emit(out, mt, Expr::Construct(mt, abs_cols));
        let mul = |s: &mut Self, out: &mut Block, m: ValueId, x: ValueId| {
            let e =
                if left { Expr::Binary(BinOp::Mul, m, x) } else { Expr::Binary(BinOp::Mul, x, m) };
            s.emit(out, ty, e)
        };
        let c = mul(self, out, mv, a.c);
        let l = (0..self.n).map(|i| mul(self, out, mv, a.l[i])).collect();
        let e = mul(self, out, am, a.e);
        let mag = self.magnitude(out, &a, vt);
        let mag = mul(self, out, am, mag);
        // Each row's products and sums, the form's parts' and the point code's: within 3 u
        // each of `|M| (|c| + rad)`.
        let k = self.k(out, ty, self.ulps((3.0, 3.0)));
        let extra = self.bin(out, BinOp::Mul, mag, k, ty);
        self.grown(out, A { c, l, e }, ty, (0.0, 0.0), Some(extra))
    }

    /// A builtin's form, or `None` where it has no rule here.
    fn builtin_rule(
        &mut self,
        out: &mut Block,
        b: Builtin,
        args: &[ValueId],
        ty: TypeId,
    ) -> R<Option<A>> {
        use Builtin as B;
        let f32 = self.m.types.f32();
        Ok(Some(match b {
            B::Abs => {
                let a = self.of_as(out, args[0], ty);
                self.abs(out, &a, ty)
            }
            B::Min | B::Max => {
                let a = self.of_as(out, args[0], ty);
                let c = self.of_as(out, args[1], ty);
                self.min_max(out, &a, &c, ty, b == B::Max)
            }
            B::Clamp => {
                let a = self.of_as(out, args[0], ty);
                let lo = self.of_as(out, args[1], ty);
                let hi = self.of_as(out, args[2], ty);
                let m = self.min_max(out, &a, &lo, ty, true);
                self.min_max(out, &m, &hi, ty, false)
            }
            B::Saturate => {
                let a = self.of_as(out, args[0], ty);
                let z = self.k(out, ty, 0.0);
                let one = self.k(out, ty, 1.0);
                let (z, one) = (self.exact(out, z), self.exact(out, one));
                let m = self.min_max(out, &a, &z, ty, true);
                self.min_max(out, &m, &one, ty, false)
            }
            B::Sqrt => {
                let a = self.of_as(out, args[0], ty);
                self.sqrt(out, &a, ty)
            }
            B::InverseSqrt => {
                let a = self.of_as(out, args[0], ty);
                let s = self.sqrt(out, &a, ty);
                let r = self.recip(out, &s, ty);
                self.rounded(out, r, ty, (4.0, 6.0))
            }
            B::Exp => {
                let a = self.of_as(out, args[0], ty);
                self.exp(out, &a, ty)
            }
            B::Dot => {
                let at = self.f.value_ty(args[0]);
                let a = self.of_as(out, args[0], at);
                let c = self.of_as(out, args[1], at);
                let p = self.mul(out, &a, &c, at);
                let s = self.sum(out, &p, at, ty);
                self.rounded(out, s, ty, (4.0, 8.0))
            }
            B::Length | B::Distance => {
                let at = self.f.value_ty(args[0]);
                let mut a = self.of_as(out, args[0], at);
                if b == B::Distance {
                    let c = self.of_as(out, args[1], at);
                    a = self.add(out, &a, &c, at, true);
                }
                if !matches!(self.m.types.get(at), TypeDef::Vector(..)) {
                    return Ok(Some(self.abs(out, &a, ty)));
                }
                let p = self.mul(out, &a, &a, at);
                let s = self.sum(out, &p, at, ty);
                let r = self.sqrt(out, &s, ty);
                // std's and WGSL's length: within a few ulps of the root of the sum.
                self.rounded(out, r, ty, (4.0, 16.0))
            }
            B::Normalize => {
                let a = self.of_as(out, args[0], ty);
                let p = self.mul(out, &a, &a, ty);
                let s = self.sum(out, &p, ty, f32);
                let l = self.sqrt(out, &s, f32);
                let r = self.recip(out, &l, f32);
                let r = self.widen_to(out, r, ty);
                let q = self.mul(out, &a, &r, ty);
                self.rounded(out, q, ty, (6.0, 12.0))
            }
            B::Mix => {
                // WGSL's: a (1 - t) + b t; std's within a few ulps of its terms of it.
                let a = self.of_as(out, args[0], ty);
                let c = self.of_as(out, args[1], ty);
                let t = self.of_as(out, args[2], ty);
                let one = self.k(out, ty, 1.0);
                let one = self.exact(out, one);
                let u = self.add(out, &one, &t, ty, true);
                let x = self.mul(out, &a, &u, ty);
                let y = self.mul(out, &c, &t, ty);
                let s = self.add(out, &x, &y, ty, false);
                let mx = self.magnitude(out, &x, ty);
                let my = self.magnitude(out, &y, ty);
                let m = self.bin(out, BinOp::Add, mx, my, ty);
                let k = self.k(out, ty, 4.0 * EPS);
                let extra = self.bin(out, BinOp::Mul, m, k, ty);
                self.grown(out, s, ty, (2.0, 2.0), Some(extra))
            }
            B::Sign | B::Floor | B::Ceil | B::Round | B::Trunc => {
                // Monotone, and exact as each target computes it: its values at the ends.
                let a = self.of_as(out, args[0], ty);
                let (lo, hi) = self.ends(out, &a, ty);
                let fl = self.builtin(out, b, vec![lo], ty);
                let fh = self.builtin(out, b, vec![hi], ty);
                self.of_ends(out, fl, fh, ty)
            }
            B::Step if !self.act.value(args[0]) => {
                // 1 where the edge is at most x: rising in x.
                let a = self.of_as(out, args[1], ty);
                let (lo, hi) = self.ends(out, &a, ty);
                let edge = self.coerce(out, args[0], ty);
                let fl = self.builtin(out, b, vec![edge, lo], ty);
                let fh = self.builtin(out, b, vec![edge, hi], ty);
                self.of_ends(out, fl, fh, ty)
            }
            B::Smoothstep if !self.act.value(args[0]) && !self.act.value(args[1]) => {
                // WGSL's: t² (3 - 2t), t = clamp((x - e0) / (e1 - e0), 0, 1), edges inactive.
                let x = self.of_as(out, args[2], ty);
                let e0 = self.coerce(out, args[0], ty);
                let e1 = self.coerce(out, args[1], ty);
                let e0f = self.exact(out, e0);
                let d = self.add(out, &x, &e0f, ty, true);
                let w = self.bin(out, BinOp::Sub, e1, e0, ty);
                let t = self.divide(out, &d, w, ty);
                let z = self.k(out, ty, 0.0);
                let one = self.k(out, ty, 1.0);
                let (z, one) = (self.exact(out, z), self.exact(out, one));
                let t = self.min_max(out, &t, &z, ty, true);
                let t = self.min_max(out, &t, &one, ty, false);
                let t2 = self.mul(out, &t, &t, ty);
                let two = self.k(out, ty, 2.0);
                let tt = self.scale(out, &t, two, ty);
                let three = self.k(out, ty, 3.0);
                let three = self.exact(out, three);
                let u = self.add(out, &three, &tt, ty, true);
                let r = self.mul(out, &t2, &u, ty);
                // A target's own way to t (a GPU may multiply by a reciprocal): ulps of t,
                // which the slope 6t(1 - t) <= 1.5 carries to the result.
                let mt = self.magnitude(out, &t, ty);
                let k = self.k(out, ty, self.ulps((2.0, 8.0)));
                let extra = self.bin(out, BinOp::Mul, mt, k, ty);
                self.grown(out, r, ty, (4.0, 8.0), Some(extra))
            }
            _ => return Ok(None),
        }))
    }

    /// The sum of a float vector form's components, as a form of `to`.
    fn sum(&mut self, out: &mut Block, a: &A, at: TypeId, to: TypeId) -> A {
        let k = match self.m.types.get(at) {
            &TypeDef::Vector(_, k) => k as u32,
            _ => return a.clone(),
        };
        let comp = |s: &mut Self, out: &mut Block, j: u32| -> A {
            let c = s.emit(out, to, Expr::Extract(a.c, j));
            let l = (0..s.n).map(|i| s.emit(out, to, Expr::Extract(a.l[i], j))).collect();
            let e = s.emit(out, to, Expr::Extract(a.e, j));
            A { c, l, e }
        };
        let mut acc = comp(self, out, 0);
        for j in 1..k {
            let x = comp(self, out, j);
            acc = self.add(out, &acc, &x, to, false);
        }
        acc
    }

    /// An operation with no rule here, by the plain interval's rule for it alone: a function of
    /// just this operation, derived by [`interval`](super::interval), called with the
    /// operands' ranges. The result keeps no slopes.
    fn by_interval(&mut self, out: &mut Block, e: &Expr, ty: TypeId) -> R<A> {
        let operands: Vec<ValueId> = match e {
            Expr::Builtin(_, a) => a.clone(),
            Expr::Binary(_, a, b) => vec![*a, *b],
            Expr::Unary(_, a) => vec![*a],
            _ => return Err(Error::not_derivable("an expression with no affine rule")),
        };
        let tys: Vec<TypeId> = operands.iter().map(|v| self.f.value_ty(*v)).collect();
        let active: Vec<bool> = operands.iter().map(|v| self.act.value(*v)).collect();
        if tys.iter().zip(&active).any(|(t, a)| *a && !is_float(&self.m.types, *t)) {
            return Err(Error::not_derivable(
                "an operation on what depends on the input with no rule",
            ));
        }
        let shape = match e {
            Expr::Builtin(b, _) => format!("{b:?}"),
            Expr::Binary(op, ..) => format!("{op:?}"),
            Expr::Unary(op, _) => format!("{op:?}"),
            _ => unreachable!("an operation"),
        };
        let key = (shape.clone(), tys.clone(), ty, active.clone());
        let w = match self.cache.affine_ops.get(&key) {
            Some(&w) => w,
            None => {
                let params = tys
                    .iter()
                    .enumerate()
                    .map(|(i, t)| Param::value(format!("x{i}"), *t))
                    .collect();
                let name = format!("affine_op_{}", shape.to_lowercase());
                let mut g = Function::new(name, params, Some(ty));
                let mut body = Vec::new();
                let xs: Vec<ValueId> = tys
                    .iter()
                    .enumerate()
                    .map(|(i, t)| g.let_(&mut body, *t, Expr::Param(i as u32)))
                    .collect();
                let op = match e {
                    Expr::Builtin(b, _) => Expr::Builtin(*b, xs.clone()),
                    Expr::Binary(op, ..) => Expr::Binary(*op, xs[0], xs[1]),
                    Expr::Unary(op, _) => Expr::Unary(*op, xs[0]),
                    _ => unreachable!("an operation"),
                };
                let r = g.let_(&mut body, ty, op);
                body.push(Stmt::Return(Some(r)));
                g.body = body;
                let w = self.m.add_function(g);
                self.cache.affine_ops.insert(key, w);
                w
            }
        };
        let mask: Vec<Bits> = tys
            .iter()
            .zip(&active)
            .map(|(t, a)| if *a { all_bits(&self.m.types, *t) } else { 0 })
            .collect();
        let spec = self.speculative > 0;
        let dg = super::interval::derive(self.m, self.cache, w, mask, self.target, spec, None)?;
        let mut args = Vec::new();
        let mut his = Vec::new();
        for (v, a) in operands.iter().zip(&active) {
            if *a {
                let t = self.f.value_ty(*v);
                let f = self.of(out, *v);
                let (lo, hi) = self.ends(out, &f, t);
                args.push(Arg::Value(lo));
                his.push(Arg::Value(hi));
            } else {
                args.push(Arg::Value(*v));
            }
        }
        args.extend(his);
        self.call_interval(out, dg, args, ty)
    }

    fn call_interval(&mut self, out: &mut Block, dg: FuncId, args: Vec<Arg>, ty: TypeId) -> R<A> {
        let rt = self.m.functions[dg.index()]
            .ret
            .ok_or_else(|| Error::internal("an interval of nothing"))?;
        let r = self.emit(out, rt, Expr::Call(dg, args));
        let lo = self.emit(out, ty, Expr::Extract(r, 0));
        let hi = self.emit(out, ty, Expr::Extract(r, 1));
        Ok(self.of_ends(out, lo, hi, ty))
    }

    /// An argument passed as parameter `by_ref` takes it: a value stored to a local first.
    fn pass(&mut self, out: &mut Block, v: ValueId, by_ref: bool) -> Arg {
        if !by_ref {
            return Arg::Value(v);
        }
        let ty = self.ty(v);
        let l = self.nf.new_local("arg", ty);
        out.push(Stmt::Store(Place::local(l), v));
        Arg::Place(Place::local(l))
    }

    /// An active argument's form: a value's, or a place's as loaded.
    fn arg_form(&mut self, out: &mut Block, a: &Arg) -> R<A> {
        match a {
            Arg::Value(x) => Ok(self.of(out, *x)),
            Arg::Place(p) => {
                let ty = place_type(self.m, self.f, p)?;
                if !self.active_root(&p.root) || !floaty(&self.m.types, ty) {
                    return Err(Error::not_derivable("an active argument that isn't a float"));
                }
                self.load_form(out, p, ty)
            }
        }
    }

    fn call(&mut self, v: Option<ValueId>, g: FuncId, args: &[Arg], out: &mut Block) -> R<()> {
        let mask: Vec<Bits> = args.iter().map(|a| arg_bits(self.m, self.f, &self.act, a)).collect();
        let known = self.m.functions[g.index()].known;
        let active_v = v.is_some_and(|v| self.act.value(v));
        if let Some(v) = v.filter(|_| active_v) {
            if known == Some(Known::ValueNoise) {
                let x = self.arg_form(out, &args[0])?;
                let a = self.noise(out, g, x)?;
                self.forms.insert(v, a);
                return Ok(());
            }
            if known == Some(Known::Smin)
                && mask[2] == 0
                && let (Arg::Value(a), Arg::Value(b), Arg::Value(k)) =
                    (&args[0], &args[1], &args[2])
            {
                let r = self.smin(out, *a, *b, *k);
                let ab: Vec<A> =
                    [a, b].iter().filter_map(|x| self.forms.get(*x).cloned()).collect();
                let f32 = self.m.types.f32();
                let r = self.unless_unbounded(out, &ab, r, f32);
                self.forms.insert(v, r);
                return Ok(());
            }
        }
        let params = self.m.functions[g.index()].params.clone();
        let dg = match derive(self.m, self.cache, g, mask.clone(), self.n, self.target) {
            Ok(dg) => dg,
            Err(Error::NotDerivable(..) | Error::ActiveLoopExit(..)) if active_v => {
                let v = v.expect("a value");
                return self.call_by_interval(out, v, g, args, &mask, &params);
            }
            Err(e) => return Err(e),
        };
        let returns = returns_active(self.m, self.cache, g, mask.clone(), Mode::Interval) != 0;
        let mut new_args = Vec::new();
        let mut extra = Vec::new();
        for ((a, bits), p) in args.iter().zip(&mask).zip(&params) {
            if *bits == 0 {
                new_args.push(a.clone());
                continue;
            }
            match a {
                // A place is lent with its slopes' and error's places, so what the callee writes
                // through a `mut` parameter reaches all of the form.
                Arg::Place(pl) if p.by_ref => {
                    let ty = place_type(self.m, self.f, pl)?;
                    if !self.active_root(&pl.root) || !floaty(&self.m.types, ty) {
                        return Err(Error::not_derivable("an active argument that isn't a float"));
                    }
                    new_args.push(a.clone());
                    extra.extend(self.place_parts(pl)?.into_iter().map(Arg::Place));
                }
                _ => {
                    let f = self.arg_form(out, a)?;
                    new_args.push(Arg::Value(f.c));
                    for x in f.l.iter().chain(std::iter::once(&f.e)) {
                        let arg = self.pass(out, *x, p.by_ref);
                        extra.push(arg);
                    }
                }
            }
        }
        new_args.extend(extra);
        match (v, returns) {
            (Some(v), true) => {
                let ty = self.f.value_ty(v);
                let rt = self.m.functions[dg.index()]
                    .ret
                    .ok_or_else(|| Error::internal("no affine type"))?;
                let r = self.emit(out, rt, Expr::Call(dg, new_args));
                let c = self.emit(out, ty, Expr::Extract(r, 0));
                let l = (0..self.n)
                    .map(|i| self.emit(out, ty, Expr::Extract(r, 1 + i as u32)))
                    .collect();
                let e = self.emit(out, ty, Expr::Extract(r, 1 + self.n as u32));
                self.forms.insert(v, A { c, l, e });
            }
            (v, _) => out.push(let_or_eval(v, Expr::Call(dg, new_args))),
        }
        Ok(())
    }

    /// A call whose callee has no affine form: its plain interval, from the arguments' ranges.
    /// Each active argument is a float or a vector (a range of an aggregate is the interval's
    /// own shape, which a form isn't turned into here); a place lent to a `mut` parameter gets
    /// back the form of the range the callee leaves in it.
    fn call_by_interval(
        &mut self,
        out: &mut Block,
        v: ValueId,
        g: FuncId,
        args: &[Arg],
        mask: &[Bits],
        params: &[Param],
    ) -> R<()> {
        let ty = self.f.value_ty(v);
        if !is_float(&self.m.types, ty) {
            return Err(Error::not_derivable("a call with no affine form that isn't a float's"));
        }
        for (a, bits) in args.iter().zip(mask) {
            let t = match a {
                Arg::Value(x) => self.f.value_ty(*x),
                Arg::Place(p) => place_type(self.m, self.f, p)?,
            };
            if *bits != 0 && !is_float(&self.m.types, t) {
                return Err(Error::not_derivable(
                    "a call with no affine form given an aggregate that depends on the input",
                ));
            }
        }
        let spec = self.speculative > 0;
        let dg =
            super::interval::derive(self.m, self.cache, g, mask.to_vec(), self.target, spec, None)?;
        let mut new_args = Vec::new();
        let mut his = Vec::new();
        // The places lent to `mut` parameters, with the locals their range is passed in.
        let mut back = Vec::new();
        let mut given = Vec::new();
        for ((a, bits), p) in args.iter().zip(mask).zip(params) {
            if *bits == 0 {
                new_args.push(a.clone());
                continue;
            }
            let f = self.arg_form(out, a)?;
            let t = self.ty(f.c);
            let (lo, hi) = self.ends(out, &f, t);
            given.push(f);
            if !p.by_ref {
                new_args.push(Arg::Value(lo));
                his.push(Arg::Value(hi));
                continue;
            }
            let (ll, hl) = (self.nf.new_local("lo", t), self.nf.new_local("hi", t));
            out.push(Stmt::Store(Place::local(ll), lo));
            out.push(Stmt::Store(Place::local(hl), hi));
            new_args.push(Arg::Place(Place::local(ll)));
            his.push(Arg::Place(Place::local(hl)));
            if let (Arg::Place(pl), true) = (a, p.mutable) {
                back.push((pl.clone(), ll, hl, t));
            }
        }
        new_args.extend(his);
        // An argument of no bound (`within`) was passed as its form's ends, which bound
        // nothing: what comes back is every number.
        let a = self.call_interval(out, dg, new_args, ty)?;
        let a = self.unless_unbounded(out, &given, a, ty);
        self.forms.insert(v, a);
        for (pl, ll, hl, t) in back {
            let lo = self.emit(out, t, Expr::Load(Place::local(ll)));
            let hi = self.emit(out, t, Expr::Load(Place::local(hl)));
            let f = self.of_ends(out, lo, hi, t);
            let f = self.unless_unbounded(out, &given, f, t);
            self.store_form(out, &pl, &f)?;
        }
        Ok(())
    }

    /// `smin(a, b, k)`, `k` inactive: `min(a, b) - max(k - |a - b|, 0)² / (4k)`, and the point
    /// code's own rounding of its steps, ulps of `|a| + |b| + k`.
    fn smin(&mut self, out: &mut Block, a: ValueId, b: ValueId, k: ValueId) -> A {
        let ty = self.m.types.f32();
        let (fa, fb) = (self.of(out, a), self.of(out, b));
        let m = self.min_max(out, &fa, &fb, ty, false);
        let d = self.add(out, &fa, &fb, ty, true);
        let h = self.abs(out, &d, ty);
        let kk = self.exact(out, k);
        let w = self.add(out, &kk, &h, ty, true);
        let z = self.k(out, ty, 0.0);
        let z = self.exact(out, z);
        let w = self.min_max(out, &w, &z, ty, true);
        let w2 = self.mul(out, &w, &w, ty);
        let least = self.k(out, ty, 1.1754944e-38);
        let kk = self.builtin(out, Builtin::Max, vec![k, least], ty);
        let quarter = self.k(out, ty, 0.25);
        let s = self.bin(out, BinOp::Div, quarter, kk, ty);
        let q = self.scale(out, &w2, s, ty);
        let r = self.add(out, &m, &q, ty, true);
        let ma = self.magnitude(out, &fa, ty);
        let mb = self.magnitude(out, &fb, ty);
        let ak = self.builtin(out, Builtin::Abs, vec![k], ty);
        let mm = self.bin(out, BinOp::Add, ma, mb, ty);
        let mm = self.bin(out, BinOp::Add, mm, ak, ty);
        let k8 = self.k(out, ty, 8.0 * EPS);
        let extra = self.bin(out, BinOp::Mul, mm, k8, ty);
        self.grown(out, r, ty, (0.0, 0.0), Some(extra))
    }

    /// `value_noise(x)`, `x` a 3-vector form: its value and gradient at the centre, the
    /// gradient through `x`'s slopes, and as error the gradient through `x`'s error, half the
    /// Hessian's bound times the reach squared (`NOISE_HESSIAN`), and the noise's own rounding.
    /// No wider than the noise's own [-1, 1].
    fn noise(&mut self, out: &mut Block, g: FuncId, x: A) -> R<A> {
        let vt = self.ty(x.c);
        let f = self.m.types.f32();
        let vg = super::ad::value_and_gradient(self.m, self.cache, g, 0, self.target, false)?;
        let rt =
            self.m.functions[vg.index()].ret.ok_or_else(|| Error::internal("no gradient type"))?;
        let by_ref = self.m.functions[vg.index()].params[0].by_ref;
        let xc = self.pass(out, x.c, by_ref);
        let r = self.emit(out, rt, Expr::Call(vg, vec![xc]));
        let val = self.emit(out, f, Expr::Extract(r, 0));
        let grad = self.emit(out, vt, Expr::Extract(r, 1));
        let l = (0..self.n)
            .map(|i| self.emit(out, f, Expr::Builtin(Builtin::Dot, vec![grad, x.l[i]])))
            .collect();
        let ag = self.builtin(out, Builtin::Abs, vec![grad], vt);
        let through = self.emit(out, f, Expr::Builtin(Builtin::Dot, vec![ag, x.e]));
        let reach = self.rad(out, &x, vt);
        let r2 = self.emit(out, f, Expr::Builtin(Builtin::Dot, vec![reach, reach]));
        let hk = self.k(out, f, 0.5 * NOISE_HESSIAN);
        let curv = self.bin(out, BinOp::Mul, r2, hk, f);
        let gr = self.emit(out, f, Expr::Builtin(Builtin::Dot, vec![ag, reach]));
        let kr = self.k(out, f, NOISE_ROUNDING);
        let gr = self.bin(out, BinOp::Mul, gr, kr, f);
        let e = self.bin(out, BinOp::Add, through, curv, f);
        let e = self.bin(out, BinOp::Add, e, gr, f);
        let e = self.bin(out, BinOp::Add, e, kr, f);
        let a = self.rounded(out, A { c: val, l, e }, f, (8.0, 16.0));
        let wide = self.rad(out, &a, f);
        let one = self.k(out, f, 1.0 + NOISE_ROUNDING);
        let z = self.zero(out, f);
        let whole = A { c: z, l: vec![z; self.n], e: one };
        let tighter = self.cmp(out, BinOp::Lt, wide, one);
        Ok(self.pick(out, tighter, &a, &whole))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `fn bump(x: mut f32) { x = x * 2.0 }` (`+ 1.0` with `add`).
    fn bump(m: &mut Module, add: bool) -> FuncId {
        let f32 = m.types.f32();
        let x = Param { name: "x".into(), ty: f32, by_ref: true, mutable: true };
        let mut g = Function::new("bump", vec![x], None);
        let mut body = Vec::new();
        let v = g.let_(&mut body, f32, Expr::Load(Place::root(PlaceRoot::Param(0))));
        let k = g.let_(&mut body, f32, Expr::Const(Const::F32(if add { 1.0 } else { 2.0 })));
        let op = if add { BinOp::Add } else { BinOp::Mul };
        let w = g.let_(&mut body, f32, Expr::Binary(op, v, k));
        body.push(Stmt::Store(Place::root(PlaceRoot::Param(0)), w));
        body.push(Stmt::Return(None));
        g.body = body;
        m.add_function(g)
    }

    /// The derived function's statements, nested blocks included, in order.
    fn flat(b: &Block, into: &mut Vec<Stmt>) {
        for s in b {
            into.push(s.clone());
            for inner in s.blocks() {
                flat(inner, into);
            }
        }
    }

    fn local_named(f: &Function, p: &Place) -> Option<String> {
        match p.root {
            PlaceRoot::Local(l) => Some(f.locals[l.index()].name.clone()),
            _ => None,
        }
    }

    /// A local lent to a `mut` parameter is lent with its slopes and error: the callee's writes
    /// reach the whole form, not copies of its parts.
    #[test]
    fn a_mut_argument_lends_its_slopes_and_error() {
        let mut m = Module::default();
        let f32 = m.types.f32();
        let g = bump(&mut m, false);
        let mut f = Function::new("doubled", vec![Param::value("q", f32)], Some(f32));
        let r = f.new_local("r", f32);
        let mut body = Vec::new();
        let q = f.let_(&mut body, f32, Expr::Param(0));
        body.push(Stmt::Store(Place::local(r), q));
        body.push(Stmt::Eval(Expr::Call(g, vec![Arg::Place(Place::local(r))])));
        let y = f.let_(&mut body, f32, Expr::Load(Place::local(r)));
        body.push(Stmt::Return(Some(y)));
        f.body = body;
        let id = m.add_function(f);
        verify(&m).expect("the source is valid IR");
        let d = derive(&mut m, &mut DeriveCache::default(), id, vec![1], 1, Target::Cpu)
            .expect("a call through a `mut` parameter has an affine form");
        verify(&m).expect("the affine form is valid IR");
        let df = &m.functions[d.index()];
        let mut all = Vec::new();
        flat(&df.body, &mut all);
        let lent: Vec<Vec<Option<String>>> = all
            .iter()
            .filter_map(|s| match s {
                Stmt::Eval(Expr::Call(_, args)) => Some(
                    args.iter()
                        .map(|a| match a {
                            Arg::Place(p) => local_named(df, p),
                            Arg::Value(_) => None,
                        })
                        .collect(),
                ),
                _ => None,
            })
            .collect();
        let want = [Some("r".to_string()), Some("r_a0".to_string()), Some("r_a1".to_string())];
        assert_eq!(lent, vec![want.to_vec()], "the call lends r's own parts");
        let called = all.iter().find_map(|s| match s {
            Stmt::Eval(Expr::Call(c, _)) => Some(*c),
            _ => None,
        });
        let callee = &m.functions[called.expect("the call is there").index()];
        assert!(callee.params.iter().all(|p| p.mutable), "the callee writes every part");
    }

    /// A branch on the input whose side writes a local through a call, or writes a `mut`
    /// parameter, restores it for the other side and joins the two after.
    #[test]
    fn a_branch_joins_what_a_call_or_a_mut_parameter_writes() {
        let mut m = Module::default();
        let f32 = m.types.f32();
        let bool_ty = m.types.bool();
        let g = bump(&mut m, true);
        // fn bumped(q: f32, y: f32) -> f32 { var r = q; if y > 0 { bump(mut r) }; r }
        let mut f = Function::new(
            "bumped",
            vec![Param::value("q", f32), Param::value("y", f32)],
            Some(f32),
        );
        let r = f.new_local("r", f32);
        let mut body = Vec::new();
        let q = f.let_(&mut body, f32, Expr::Param(0));
        let y = f.let_(&mut body, f32, Expr::Param(1));
        body.push(Stmt::Store(Place::local(r), q));
        let z = f.let_(&mut body, f32, Expr::Const(Const::F32(0.0)));
        let c = f.let_(&mut body, bool_ty, Expr::Binary(BinOp::Gt, y, z));
        let then = vec![Stmt::Eval(Expr::Call(g, vec![Arg::Place(Place::local(r))]))];
        body.push(Stmt::If { cond: c, then, else_: Vec::new() });
        let v = f.let_(&mut body, f32, Expr::Load(Place::local(r)));
        body.push(Stmt::Return(Some(v)));
        f.body = body;
        let id = m.add_function(f);
        // fn bump_if(x: mut f32, y: f32) { if y > 0 { x = x + 1 } }
        let x = Param { name: "x".into(), ty: f32, by_ref: true, mutable: true };
        let mut h = Function::new("bump_if", vec![x, Param::value("y", f32)], None);
        let mut body = Vec::new();
        let y = h.let_(&mut body, f32, Expr::Param(1));
        let z = h.let_(&mut body, f32, Expr::Const(Const::F32(0.0)));
        let c = h.let_(&mut body, bool_ty, Expr::Binary(BinOp::Gt, y, z));
        let mut then = Vec::new();
        let v = h.let_(&mut then, f32, Expr::Load(Place::root(PlaceRoot::Param(0))));
        let one = h.let_(&mut then, f32, Expr::Const(Const::F32(1.0)));
        let w = h.let_(&mut then, f32, Expr::Binary(BinOp::Add, v, one));
        then.push(Stmt::Store(Place::root(PlaceRoot::Param(0)), w));
        body.push(Stmt::If { cond: c, then, else_: Vec::new() });
        body.push(Stmt::Return(None));
        h.body = body;
        let hid = m.add_function(h);
        verify(&m).expect("the source is valid IR");
        let mut cache = DeriveCache::default();
        let d = derive(&mut m, &mut cache, id, vec![1, 1], 2, Target::Cpu)
            .expect("a branch around a call has an affine form");
        let dh = derive(&mut m, &mut cache, hid, vec![1, 1], 2, Target::Cpu)
            .expect("a branch writing a `mut` parameter has an affine form");
        verify(&m).expect("the affine forms are valid IR");
        // After the call, r is stored again: restored for the other side, then joined.
        let df = &m.functions[d.index()];
        let mut all = Vec::new();
        flat(&df.body, &mut all);
        let call = all
            .iter()
            .position(|s| matches!(s, Stmt::Eval(Expr::Call(..))))
            .expect("the call is derived");
        let stores_r = all[call..]
            .iter()
            .filter(
                |s| matches!(s, Stmt::Store(p, _) if local_named(df, p).as_deref() == Some("r")),
            )
            .count();
        assert!(stores_r >= 2, "r is restored and joined after the call ({stores_r} stores)");
        // The parameter is stored by the side, restored, and joined: three times, not once.
        let dhf = &m.functions[dh.index()];
        let mut all = Vec::new();
        flat(&dhf.body, &mut all);
        let stores_x = all
            .iter()
            .filter(|s| matches!(s, Stmt::Store(p, _) if p.root == PlaceRoot::Param(0)))
            .count();
        assert!(stores_x >= 3, "x is restored and joined ({stores_x} stores)");
    }

    /// A call whose callee has no affine form, given an aggregate that depends on the input,
    /// leaves the whole function to the plain interval rather than bounding the aggregate as a
    /// float.
    #[test]
    fn an_aggregate_given_to_a_call_with_no_form_leaves_the_interval() {
        let mut m = Module::default();
        let f32 = m.types.f32();
        let pair = m.types.intern(TypeDef::Struct {
            name: "Pair".into(),
            fields: vec![("a".into(), f32), ("b".into(), f32)],
        });
        let range = m.types.intern(TypeDef::Struct {
            name: "Range".into(),
            fields: vec![("lo".into(), f32), ("hi".into(), f32)],
        });
        // fn g(p: Pair) -> f32 { f32(i32(p.a)) }: a conversion, which has no affine rule.
        let mut g = Function::new("g", vec![Param::value("p", pair)], Some(f32));
        let mut body = Vec::new();
        let p = g.let_(&mut body, pair, Expr::Param(0));
        let a = g.let_(&mut body, f32, Expr::Extract(p, 0));
        let i32t = m.types.scalar(Scalar::I32);
        let i = g.let_(&mut body, i32t, Expr::Convert(a, Scalar::I32));
        let back = g.let_(&mut body, f32, Expr::Convert(i, Scalar::F32));
        body.push(Stmt::Return(Some(back)));
        g.body = body;
        let gid = m.add_function(g);
        let mut f = Function::new("f", vec![Param::value("q", f32)], Some(f32));
        let mut body = Vec::new();
        let q = f.let_(&mut body, f32, Expr::Param(0));
        let p = f.let_(&mut body, pair, Expr::Construct(pair, vec![q, q]));
        let r = f.let_(&mut body, f32, Expr::Call(gid, vec![Arg::Value(p)]));
        body.push(Stmt::Return(Some(r)));
        f.body = body;
        let id = m.add_function(f);
        verify(&m).expect("the source is valid IR");
        affine(&mut m, &mut DeriveCache::default(), id, 0, Target::Cpu, (range, range))
            .expect("the plain interval answers");
        verify(&m).expect("no form of an aggregate is bounded as a float");
    }
}
