//! Forward-mode differentiation (D-012). An active value of type `T` carries `n` tangents, also
//! of type `T`: its derivative along each of the input's `n` components. A derived function takes
//! a tangent for each active parameter (appended after the originals, `n` per parameter) and,
//! when its result is active, returns `{value, d0, ..., d(n-1)}`.

use super::*;
use std::collections::HashMap;

fn dims(m: &Module, t: TypeId) -> Result<u8> {
    match m.types.get(t) {
        TypeDef::Scalar(Scalar::F32) => Ok(1),
        TypeDef::Vector(Scalar::F32, n) => Ok(*n),
        _ => Err(Error::not_derivable(format!(
            "a gradient needs an `f32` or a float vector, not `{}`",
            m.types.display(t)
        ))),
    }
}

pub(super) fn value_and_gradient(
    m: &mut Module,
    cache: &mut DeriveCache,
    f: FuncId,
    ncap: u32,
    target: Target,
    with: bool,
) -> Result<FuncId> {
    let func = m.functions[f.index()].derived_signature();
    let xi = ncap as usize;
    let x_ty = func.params.get(xi).ok_or_else(|| Error::internal("no input parameter"))?.ty;
    let n = dims(m, x_ty)?;
    let f32 = m.types.f32();
    // With a payload: `f` returns `(f32, T)`; the value is its first field. A `T` with no value
    // (`()`) leaves the value alone in it.
    let (wrapped, payload) = match (with, func.ret.map(|r| m.types.get(r).clone())) {
        (false, _) if func.ret == Some(f32) && !func.ret_ref => (false, None),
        (true, Some(TypeDef::Struct { fields, .. }))
            if fields.len() == 2 && fields[0].1 == f32 && !func.ret_ref =>
        {
            (true, Some(fields[1].1))
        }
        (true, Some(TypeDef::Struct { fields, .. }))
            if fields.len() == 1 && fields[0].1 == f32 && !func.ret_ref =>
        {
            (true, None)
        }
        (false, _) => {
            return Err(Error::not_derivable("a gradient needs a function that returns an `f32`"));
        }
        (true, _) => {
            return Err(Error::not_derivable(
                "`value_gradient_with` needs a function that returns `(f32, T)`",
            ));
        }
    };
    let mut mask = vec![0; func.params.len()];
    mask[xi] = all_bits(&m.types, x_ty);
    let d = derive(m, cache, f, mask.clone(), n, target, Variant::Point)?;
    let returns = returns_active(m, cache, f, mask, Mode::Ad) != 0;
    let tuple = value_gradient_ty(&mut m.types, x_ty, payload);
    let mut g = Function::new(
        format!("{}_value_and_gradient", func.name),
        func.params.clone(),
        Some(tuple),
    );
    let mut body = Vec::new();
    let args: Vec<Arg> = (0..func.params.len() as u32).map(|i| g.param_arg(i, &mut body)).collect();
    // Seeds: the unit vectors along each component of x.
    let r = call_seeded(m, &mut g, &mut body, d, args, &func.params[xi], n, |g, body, k| {
        let one = g.let_(body, f32, Expr::Const(Const::F32(1.0)));
        if n == 1 {
            return one;
        }
        let zero = g.let_(body, f32, Expr::Const(Const::F32(0.0)));
        let comps = (0..n).map(|c| if c == k { one } else { zero }).collect();
        g.let_(body, x_ty, Expr::Construct(x_ty, comps))
    })?;
    // The value (with a payload, the first field of `f`'s result), and the payload.
    let ret = func.ret.ok_or_else(|| Error::internal("no result"))?;
    let first = |g: &mut Function, body: &mut Block, of: ValueId| {
        if wrapped { g.let_(body, f32, Expr::Extract(of, 0)) } else { of }
    };
    let (value, parts, result) = if returns {
        let whole = g.let_(&mut body, ret, Expr::Extract(r, 0));
        let v = first(&mut g, &mut body, whole);
        let mut parts = Vec::new();
        for k in 0..n {
            let t = g.let_(&mut body, ret, Expr::Extract(r, 1 + k as u32));
            parts.push(first(&mut g, &mut body, t));
        }
        (v, parts, whole)
    } else {
        // The function doesn't depend on its input: the gradient is zero.
        let z = g.let_(&mut body, f32, Expr::Const(Const::F32(0.0)));
        let v = first(&mut g, &mut body, r);
        (v, vec![z; n as usize], r)
    };
    let grad =
        if n == 1 { parts[0] } else { g.let_(&mut body, x_ty, Expr::Construct(x_ty, parts)) };
    let mut items = vec![value, grad];
    if let Some(t) = payload {
        items.push(g.let_(&mut body, t, Expr::Extract(result, 1)));
    }
    let out = g.let_(&mut body, tuple, Expr::Construct(tuple, items));
    body.push(Stmt::Return(Some(out)));
    g.body = body;
    Ok(m.add_function(g))
}

/// Calls `d`, a derivation of a function by its parameter `x` (`n` components), in `g`: with
/// `args`, then a seed for each direction (`seed` makes direction `k`'s), passed the way `x` is
/// (a tangent parameter is passed like its original). The call's result.
#[allow(clippy::too_many_arguments)]
fn call_seeded(
    m: &Module,
    g: &mut Function,
    body: &mut Block,
    d: FuncId,
    mut args: Vec<Arg>,
    x: &Param,
    n: u8,
    mut seed: impl FnMut(&mut Function, &mut Block, u8) -> ValueId,
) -> Result<ValueId> {
    for k in 0..n {
        let s = seed(g, body, k);
        if x.by_ref {
            let l = g.new_local(format!("seed{k}"), x.ty);
            body.push(Stmt::Store(Place::local(l), s));
            args.push(Arg::Place(Place::local(l)));
        } else {
            args.push(Arg::Value(s));
        }
    }
    let dual_ty = m.functions[d.index()]
        .ret
        .ok_or_else(|| Error::internal("derived function returns nothing"))?;
    Ok(g.let_(body, dual_ty, Expr::Call(d, args)))
}

/// Which derivative of a function a derivation writes, part of the cache's key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Variant {
    /// The derivative at a point.
    Point,
    /// A filtering one ([`Ad::filter`]).
    Filtered,
    /// One whose code only an interval reads ([`Ad::bounds`]).
    Bounded,
}

impl Variant {
    /// What the derived function's name says it is, before its directions.
    fn suffix(self) -> &'static str {
        match self {
            Variant::Point => "d",
            Variant::Filtered => "f",
            Variant::Bounded => "b",
        }
    }
}

/// A footprint narrower than this is a point: the builtin's own value is kept.
const NARROWEST: f64 = 1.0e-6;

/// The builtins a filtering derivation box-filters: those that jump, kink, oscillate, or transition
/// steeply (`fract`..`saturate`, `sin`, `cos`, `smoothstep`), where a point sample aliases.
fn filters(b: Builtin) -> bool {
    use Builtin as B;
    matches!(
        b,
        B::Fract
            | B::Floor
            | B::Ceil
            | B::Round
            | B::Trunc
            | B::Step
            | B::Sign
            | B::Abs
            | B::Min
            | B::Max
            | B::Clamp
            | B::Saturate
            | B::Sin
            | B::Cos
            | B::Smoothstep
    )
}

/// A function of `f`'s captures, `x` and `width` (of `x`'s type) returning `f`'s value at `x`
/// box-filtered over a footprint `width` wide along each of `x`'s components (`filtered`,
/// language.md §14): forward-mode differentiation with each direction's seed its width, so
/// every value's tangents span its range over the footprint (to first order), and each
/// aliasing builtin or lattice noise is replaced by its filtered average over that range.
pub(super) fn filtered_value(
    m: &mut Module,
    cache: &mut DeriveCache,
    f: FuncId,
    ncap: u32,
    target: Target,
) -> Result<FuncId> {
    let func = m.functions[f.index()].derived_signature();
    let xi = ncap as usize;
    let x_ty = func.params.get(xi).ok_or_else(|| Error::internal("no input parameter"))?.ty;
    let n = dims(m, x_ty)?;
    let f32 = m.types.f32();
    if func.ret_ref {
        return Err(Error::not_derivable(
            "`filtered` needs a function that returns a value, not a reference",
        ));
    }
    let Some(ret_ty) = func.ret else {
        return Err(Error::not_derivable("`filtered` needs a function that returns a value"));
    };
    let mut mask = vec![0; func.params.len()];
    mask[xi] = all_bits(&m.types, x_ty);
    let d = derive(m, cache, f, mask.clone(), n, target, Variant::Filtered)?;
    let returns = returns_active(m, cache, f, mask, Mode::Ad) != 0;
    let mut params = func.params.clone();
    let width_param = params.len() as u32;
    params.push(Param::value("width", x_ty));
    let mut g = Function::new(format!("{}_filtered", func.name), params, Some(ret_ty));
    let mut body = Vec::new();
    let args: Vec<Arg> = (0..func.params.len() as u32).map(|i| g.param_arg(i, &mut body)).collect();
    let width = g.let_(&mut body, x_ty, Expr::Param(width_param));
    // Seeds: each component of x's axis, its width long.
    let r = call_seeded(m, &mut g, &mut body, d, args, &func.params[xi], n, |g, body, k| {
        if n == 1 {
            return width;
        }
        let zero = g.let_(body, f32, Expr::Const(Const::F32(0.0)));
        let wk = g.let_(body, f32, Expr::Extract(width, k as u32));
        let comps = (0..n).map(|c| if c == k { wk } else { zero }).collect();
        g.let_(body, x_ty, Expr::Construct(x_ty, comps))
    })?;
    let value = if returns { g.let_(&mut body, ret_ty, Expr::Extract(r, 0)) } else { r };
    body.push(Stmt::Return(Some(value)));
    g.body = body;
    Ok(m.add_function(g))
}

/// The derivative of `f` with active parameters `mask` and `n` directions, the `variant` asked
/// for.
pub(super) fn derive(
    m: &mut Module,
    cache: &mut DeriveCache,
    f: FuncId,
    mask: Vec<Bits>,
    n: u8,
    target: Target,
    variant: Variant,
) -> Result<FuncId> {
    if let Some(&d) = cache.ad.get(&(f, mask.clone(), n, variant)) {
        return Ok(d);
    }
    let (filter, bounds) = (variant == Variant::Filtered, variant == Variant::Bounded);
    let func = m.functions[f.index()].clone();
    no_gpu_or_heap_work(&func.body)?;
    let returns = returns_active(m, cache, f, mask.clone(), Mode::Ad) != 0;
    let act = body_activity(m, cache, f, &mask, Mode::Ad);
    no_heap_store(&act)?;
    no_write_through_mut(&func, &act, &mask, "a derivative")?;
    let mut nf = func.derived_signature();
    nf.name = format!("{}_{}{}", func.name, variant.suffix(), n);
    // Tangent parameters, appended: n per active parameter.
    let mut tparam: Vec<Vec<u32>> = vec![Vec::new(); func.params.len()];
    for (i, p) in func.params.iter().enumerate() {
        if mask[i] != 0 {
            for k in 0..n {
                tparam[i].push(nf.params.len() as u32);
                nf.params.push(Param {
                    name: format!("{}_d{k}", p.name),
                    ty: p.ty,
                    by_ref: p.by_ref,
                    mutable: p.mutable,
                });
            }
        }
    }
    let mut tlocal: Vec<Vec<LocalId>> = vec![Vec::new(); func.locals.len()];
    for (l, decl) in func.locals.iter().enumerate() {
        if act.locals[l] != 0 {
            for k in 0..n {
                tlocal[l].push(nf.new_local(format!("{}_d{k}", decl.name), decl.ty));
            }
        }
    }
    if returns {
        let r = func.ret.ok_or_else(|| Error::internal("a missing result type"))?;
        // A projection's: a pointer into the place, and one into each tangent's.
        let r = if func.ret_ref { m.types.intern(TypeDef::Ptr(r)) } else { r };
        nf.ret_ref = false;
        let mut fields = vec![("v".to_string(), r)];
        for k in 0..n {
            fields.push((format!("d{k}"), r));
        }
        nf.ret =
            Some(m.types.intern(TypeDef::Struct {
                name: format!("dual{n}<{}>", m.types.display(r)),
                fields,
            }));
    }
    // With its signature, for a recursive call to read.
    let id = m.add_function(Function::new(nf.name.clone(), nf.params.clone(), nf.ret));
    // Cached before its body is built, for calls back into it; removed again if building
    // fails, so a later request doesn't find the bodiless placeholder.
    let key = (f, mask, n, variant);
    cache.ad.insert(key.clone(), id);
    cache.ad_of.insert(id, key.clone());
    // What only a bounded derivation reads (`sum_of_squares`).
    let (defs, single, scope) = if bounds {
        let single = super::interval::single_stores(m, &func);
        (crate::visit::definitions(&func.body), single, vec![false; func.values.len()])
    } else {
        Default::default()
    };
    let mut b = Ad {
        m,
        cache,
        f: &func,
        nf,
        act,
        n,
        target,
        filter,
        bounds,
        defs,
        single,
        scope,
        tparam,
        tlocal,
        tv: HashMap::new(),
        returns,
        recips: HashMap::new(),
        made: Vec::new(),
        scaled: HashMap::new(),
        track: Track::default(),
    };
    let body = match b.block(&func.body) {
        Ok(body) => body,
        Err(e) => {
            b.cache.ad.remove(&key);
            b.cache.ad_of.remove(&id);
            return Err(e);
        }
    };
    let mut nf = b.nf;
    nf.body = body;
    m.functions[id.index()] = nf;
    Ok(id)
}

struct Ad<'a> {
    m: &'a mut Module,
    cache: &'a mut DeriveCache,
    f: &'a Function,
    nf: Function,
    act: Activity,
    n: u8,
    target: Target,
    /// Filtering (`filtered`): the tangents are each direction's derivative times the
    /// footprint's width along it, and the discontinuous builtins are box-filtered over the
    /// footprint they give (see `filtered_let`).
    filter: bool,
    /// Code only an interval reads (`interval` of a `gradient`), so written for its interval to
    /// be tight, not for its point's rounding: the square root of a sum of squares, `sqrt(c Σ
    /// tᵢ²)` (a length written out, as a round cone's is), is differentiated as `length` is,
    /// `√c dot(t / |t|, dt)`, whose interval is bounded where `t` can be 0, where the quotient
    /// `d(Σ tᵢ²) / 2√(Σ tᵢ²)` is every number. The point code is not changed.
    bounds: bool,
    /// The primal's values, by the expression each is (`bounds` only).
    defs: HashMap<ValueId, &'a Expr>,
    /// By local: the one value stored in it, if a `let`'s (`bounds` only).
    single: Vec<Option<ValueId>>,
    /// By value: whether it's defined in the blocks being derived, what's in scope (`bounds`
    /// only).
    scope: Vec<bool>,
    tparam: Vec<Vec<u32>>,
    tlocal: Vec<Vec<LocalId>>,
    /// Each active value's tangents (`None`: structurally zero).
    tv: HashMap<ValueId, Vec<Option<ValueId>>>,
    returns: bool,
    /// Reciprocals made while differentiating one statement, shared by its `n` directions:
    /// one division and `n` multiplications instead of `n` divisions.
    recips: HashMap<ValueId, ValueId>,
    /// The expressions made while differentiating one statement, so what its directions share
    /// (`2v` for a square root, a clamp's masks) is made once.
    made: Vec<(TypeId, Expr, ValueId)>,
    /// Tangents made as another tangent times a factor every direction shares (see `scale`).
    scaled: HashMap<ValueId, (ValueId, ValueId)>,
    track: Track,
}

impl Builder for Ad<'_> {
    fn emit(&mut self, out: &mut Block, ty: TypeId, e: Expr) -> ValueId {
        if let Some((_, _, v)) = self.made.iter().find(|(t, x, _)| *t == ty && *x == e) {
            return *v;
        }
        let v = self.nf.new_value(ty);
        out.push(Stmt::Let(v, e.clone()));
        if !matches!(e, Expr::Call(..) | Expr::Host(..)) {
            self.made.push((ty, e, v));
        }
        v
    }

    fn types(&self) -> &Types {
        &self.m.types
    }

    fn ty(&self, v: ValueId) -> TypeId {
        self.nf.value_ty(v)
    }
}

impl Ad<'_> {
    /// Which derivative this derivation writes.
    fn variant(&self) -> Variant {
        if self.filter {
            Variant::Filtered
        } else if self.bounds {
            Variant::Bounded
        } else {
            Variant::Point
        }
    }

    /// A float constant of `ty`'s element type (`f32` or `f64`), as a scalar: operations on
    /// `ty` splat it.
    fn fc(&mut self, out: &mut Block, ty: TypeId, x: f64) -> ValueId {
        let (t, c) = match self.m.types.element_scalar(ty) {
            Some(Scalar::F64) => (self.m.types.scalar(Scalar::F64), Const::F64(x)),
            Some(Scalar::F16) => (self.m.types.scalar(Scalar::F16), Const::F16(x as f32)),
            _ => (self.m.types.f32(), Const::F32(x as f32)),
        };
        self.emit(out, t, Expr::Const(c))
    }

    fn t(&self, v: ValueId, k: usize) -> Option<ValueId> {
        self.tv.get(&v).and_then(|ts| ts[k])
    }

    fn is_matrix(&self, v: ValueId) -> bool {
        matches!(self.m.types.get(self.ty(v)), TypeDef::Matrix(..))
    }

    /// The tangent `t` as an operand of type `ty`, or a zero for a structurally zero one.
    fn tangent_or_zero(&mut self, out: &mut Block, t: Option<ValueId>, ty: TypeId) -> ValueId {
        match t {
            Some(t) => self.coerce(out, t, ty),
            None => self.zero(out, ty),
        }
    }

    /// The tangent `t` times `f`, a factor every direction shares. A tangent made that way,
    /// `b × g`, becomes `b × (g f)`, with `g f` made once for all directions: a chain of
    /// factors (`y y l2`, `sqrt(s) / l2`) costs one multiplication per direction, not one each.
    fn scale(&mut self, out: &mut Block, t: ValueId, f: ValueId, ty: TypeId) -> ValueId {
        if self.is_matrix(t) || self.is_matrix(f) {
            return self.bin(out, BinOp::Mul, t, f, ty);
        }
        if let Some(&(b, g)) = self.scaled.get(&t)
            && self.ty(g) == self.ty(f)
        {
            let gt = self.ty(g);
            let gf = self.bin(out, BinOp::Mul, g, f, gt);
            let r = self.bin(out, BinOp::Mul, b, gf, ty);
            self.scaled.insert(r, (b, gf));
            return r;
        }
        let r = self.bin(out, BinOp::Mul, t, f, ty);
        self.scaled.insert(r, (t, f));
        r
    }

    /// `x / d` as `x × (1 / d)`, with `1 / d` made once per statement.
    fn div(&mut self, out: &mut Block, x: ValueId, d: ValueId, ty: TypeId) -> ValueId {
        let r = match self.recips.get(&d) {
            Some(&r) => r,
            None => {
                let dt = self.ty(d);
                let one = self.fc(out, dt, 1.0);
                let one = self.coerce(out, one, dt);
                let r = self.bin(out, BinOp::Div, one, d, dt);
                self.recips.insert(d, r);
                r
            }
        };
        self.scale(out, x, r, ty)
    }

    /// `x / d` for a vector's length `d` or twice a square root, with `d` raised to at least
    /// 10⁻³⁰. Where it's zero, `x` is often zero too: inside a box, `length(max(q, 0))` is zero
    /// and moves in no direction, and `sqrt(max(1 - t², 0))` is zero for every |t| > 1. Their
    /// tangents are then zero, not 0/0; and where `x` isn't zero, the tangent is large but
    /// finite, so a mask of 0 that doesn't take it (`clamp`, a vector's `max`) gives 0, not
    /// ∞ × 0. One `max` per statement, shared by its directions.
    fn div_from_zero(&mut self, out: &mut Block, x: ValueId, d: ValueId, ty: TypeId) -> ValueId {
        let dt = self.ty(d);
        let floor = self.fc(out, dt, 1e-30);
        let d = self.builtin(out, Builtin::Max, vec![d, floor], dt);
        self.div(out, x, d, ty)
    }

    /// `x + y` where either may be zero.
    fn add(
        &mut self,
        out: &mut Block,
        x: Option<ValueId>,
        y: Option<ValueId>,
        ty: TypeId,
    ) -> Option<ValueId> {
        match (x, y) {
            (None, None) => None,
            (Some(a), None) | (None, Some(a)) => Some(a),
            (Some(a), Some(b)) => Some(self.bin(out, BinOp::Add, a, b, ty)),
        }
    }

    fn sub(
        &mut self,
        out: &mut Block,
        x: Option<ValueId>,
        y: Option<ValueId>,
        ty: TypeId,
    ) -> Option<ValueId> {
        match (x, y) {
            (None, None) => None,
            (Some(a), None) => Some(a),
            (None, Some(b)) => {
                let b = self.coerce(out, b, ty);
                Some(self.emit(out, ty, Expr::Unary(UnOp::Neg, b)))
            }
            (Some(a), Some(b)) => Some(self.bin(out, BinOp::Sub, a, b, ty)),
        }
    }

    /// `x * y` where `x` may be zero.
    fn mul(
        &mut self,
        out: &mut Block,
        x: Option<ValueId>,
        y: ValueId,
        ty: TypeId,
    ) -> Option<ValueId> {
        x.map(|x| self.scale(out, x, y, ty))
    }

    fn place(&self, p: &Place, k: usize) -> Result<Place> {
        let root = match &p.root {
            PlaceRoot::Local(l) => PlaceRoot::Local(
                *self.tlocal[l.index()]
                    .get(k)
                    .ok_or_else(|| Error::internal("an inactive local's tangent"))?,
            ),
            PlaceRoot::Param(i) => PlaceRoot::Param(
                *self.tparam[*i as usize]
                    .get(k)
                    .ok_or_else(|| Error::internal("an inactive parameter's tangent"))?,
            ),
            // A projection's tangent points into the place's tangent.
            PlaceRoot::Ptr(v) => PlaceRoot::Ptr(
                self.t(*v, k).ok_or_else(|| Error::internal("an inactive pointer's tangent"))?,
            ),
            _ => return Err(Error::not_derivable("can't differentiate through this place")),
        };
        Ok(Place { root, path: p.path.clone() })
    }

    fn zero(&mut self, out: &mut Block, ty: TypeId) -> ValueId {
        self.emit(out, ty, Expr::Zero(ty))
    }

    fn block(&mut self, b: &Block) -> Result<Block> {
        let mut out = Vec::new();
        for s in b {
            self.stmt(s, &mut out)?;
            if let Stmt::Let(v, _) = s
                && let Some(x) = self.scope.get_mut(v.index())
            {
                *x = true;
            }
        }
        // The block's values go out of scope with it.
        for s in b {
            if let Stmt::Let(v, _) = s
                && let Some(x) = self.scope.get_mut(v.index())
            {
                *x = false;
            }
        }
        Ok(out)
    }

    /// `v` as `c Σ tᵢ²` (`c` absent for 1): the terms (scalars and vectors, each a value in
    /// scope) and the factor, which depends on nothing derived. Through `+`, `x * x`, `dot(x,
    /// x)` and `let` bindings.
    fn sum_of_squares(&self, v: ValueId) -> Option<(Vec<ValueId>, Option<ValueId>)> {
        if let Expr::Binary(BinOp::Mul, a, b) = *self.def(v)? {
            for (x, c) in [(a, b), (b, a)] {
                if !self.act.value(c) && self.m.types.as_scalar(self.ty(c)).is_some() {
                    if let Some(t) = self.squares(x) {
                        return Some((t, Some(c)));
                    }
                }
            }
        }
        self.squares(v).map(|t| (t, None))
    }

    fn squares(&self, v: ValueId) -> Option<Vec<ValueId>> {
        match *self.def(v)? {
            Expr::Binary(BinOp::Add, a, b) => {
                let mut t = self.squares(a)?;
                t.extend(self.squares(b)?);
                Some(t)
            }
            Expr::Binary(BinOp::Mul, a, b) if self.track.same_value(a, b) || a == b => {
                self.in_scope(a).then(|| vec![a])
            }
            Expr::Builtin(Builtin::Dot, ref args)
                if args.len() == 2
                    && (args[0] == args[1] || self.track.same_value(args[0], args[1])) =>
            {
                self.in_scope(args[0]).then(|| vec![args[0]])
            }
            _ => None,
        }
    }

    /// What `v` is, through a load of a `let`'s local.
    fn def(&self, v: ValueId) -> Option<&Expr> {
        let e = *self.defs.get(&v)?;
        let Expr::Load(p) = e else { return Some(e) };
        let Some(l) = p.whole_local() else { return Some(e) };
        let src = self.single.get(l.index()).copied().flatten()?;
        if self.in_scope(src) { self.def(src) } else { None }
    }

    fn in_scope(&self, v: ValueId) -> bool {
        self.scope.get(v.index()).copied().unwrap_or(false)
    }

    /// The tangent along direction `k` of `sqrt(c Σ tᵢ²)` as `length`'s rule gives it (see
    /// [`Ad::bounds`]): `None` where the terms aren't 2 to 4 floats in all.
    fn root_of_squares(
        &mut self,
        terms: &[ValueId],
        factor: Option<ValueId>,
        k: usize,
        ty: TypeId,
        out: &mut Block,
    ) -> Option<ValueId> {
        let mut comps = Vec::new();
        let mut dcomps = Vec::new();
        let zero = self.fc(out, ty, 0.0);
        for &t in terms {
            let tt = self.ty(t);
            let n = match *self.m.types.get(tt) {
                TypeDef::Scalar(Scalar::F32) => 1,
                TypeDef::Vector(Scalar::F32, n) => n,
                _ => return None,
            };
            let dt = self.t(t, k);
            for c in 0..n as u32 {
                let (x, dx) = if n == 1 {
                    (t, dt.unwrap_or(zero))
                } else {
                    let x = self.emit(out, ty, Expr::Extract(t, c));
                    let dx = match dt {
                        Some(d) => self.emit(out, ty, Expr::Extract(d, c)),
                        None => zero,
                    };
                    (x, dx)
                };
                comps.push(x);
                dcomps.push(dx);
            }
        }
        if !(2..=4).contains(&comps.len()) {
            return None;
        }
        let vt = self.m.types.vector_of(Scalar::F32, comps.len() as u8);
        let w = self.emit(out, vt, Expr::Construct(vt, comps.clone()));
        let dw = self.emit(out, vt, Expr::Construct(vt, dcomps));
        // As `length`'s: `normalize` where `w` isn't 0, 0 where it is.
        let len = self.builtin(out, Builtin::Length, vec![w], ty);
        let b = self.m.types.bool();
        let pos = self.emit(out, b, Expr::Binary(BinOp::Gt, len, zero));
        let one = self.fc(out, ty, 1.0);
        let mut axis = vec![one];
        axis.extend((1..comps.len()).map(|_| zero));
        let axis = self.emit(out, vt, Expr::Construct(vt, axis));
        let safe = self.emit(out, vt, Expr::Select { cond: pos, if_true: w, if_false: axis });
        let unit = self.builtin(out, Builtin::Normalize, vec![safe], vt);
        let none = self.emit(out, vt, Expr::Zero(vt));
        let dir = self.emit(out, vt, Expr::Select { cond: pos, if_true: unit, if_false: none });
        let d = self.builtin(out, Builtin::Dot, vec![dir, dw], ty);
        Some(match factor {
            Some(c) => {
                let r = self.builtin(out, Builtin::Sqrt, vec![c], ty);
                self.bin(out, BinOp::Mul, d, r, ty)
            }
            None => d,
        })
    }

    fn stmt(&mut self, s: &Stmt, out: &mut Block) -> Result<()> {
        // What one statement shares among its directions stays within it.
        self.made.clear();
        self.recips.clear();
        self.track.before(s);
        let r = self.stmt_inner(s, out);
        self.track.after(s, r)
    }

    fn stmt_inner(&mut self, s: &Stmt, out: &mut Block) -> Result<()> {
        match s {
            Stmt::Let(v, Expr::Call(g, args)) => self.call(Some(*v), *g, args, out),
            Stmt::Eval(Expr::Call(g, args)) => self.call(None, *g, args, out),
            Stmt::Let(v, e) => {
                match e {
                    Expr::Builtin(b, args) if self.filter && self.act.value(*v) && filters(*b) => {
                        self.filtered_let(*v, *b, args, out)
                    }
                    _ => out.push(s.clone()),
                }
                if self.act.value(*v) {
                    let mut ts = Vec::new();
                    for k in 0..self.n as usize {
                        ts.push(self.tangent(*v, e, k, out)?);
                    }
                    self.tv.insert(*v, ts);
                }
                Ok(())
            }
            Stmt::Store(p, v) => {
                out.push(s.clone());
                if self.act.place(self.m, self.f, p) && !matches!(p.root, PlaceRoot::Resource(_)) {
                    let ty = place_type(self.m, self.f, p)?;
                    for k in 0..self.n as usize {
                        let tp = self.place(p, k)?;
                        let tv = self.tangent_or_zero(out, self.t(*v, k), ty);
                        out.push(Stmt::Store(tp, tv));
                    }
                }
                Ok(())
            }
            Stmt::Return(Some(v)) if self.returns => {
                let r = self.ty(*v);
                let mut parts = vec![*v];
                for k in 0..self.n as usize {
                    let t = self.tangent_or_zero(out, self.t(*v, k), r);
                    parts.push(t);
                }
                let dual = self.nf.ret.ok_or_else(|| Error::internal("a missing result type"))?;
                let d = self.emit(out, dual, Expr::Construct(dual, parts));
                out.push(Stmt::Return(Some(d)));
                Ok(())
            }
            other => pass_through(other, out, &mut |b| self.block(b)),
        }
    }

    fn call(&mut self, v: Option<ValueId>, g: FuncId, args: &[Arg], out: &mut Block) -> Result<()> {
        let mask: Vec<Bits> = args.iter().map(|a| arg_bits(self.m, self.f, &self.act, a)).collect();
        if mask.iter().all(|b| *b == 0) {
            out.push(let_or_eval(v, Expr::Call(g, args.to_vec())));
            return Ok(());
        }
        let returns = returns_active(self.m, self.cache, g, mask.clone(), Mode::Ad) != 0;
        // A call that gives back nothing a derivative flows through (its result carries none,
        // and it writes no active argument) runs as it is: `skeleton.find(name)` reads an
        // active skeleton for a bone's index, through code (comparing text) that has no
        // derivative.
        let params = &self.m.functions[g.index()].params;
        let writes = mask.iter().zip(params).any(|(b, p)| *b != 0 && p.mutable);
        if !returns && !writes {
            out.push(let_or_eval(v, Expr::Call(g, args.to_vec())));
            return Ok(());
        }
        if self.filter
            && let (Some(v), true) = (v, returns)
            && self.m.functions[g.index()].known == Some(Known::ValueNoise)
            && args.len() == 1
        {
            // Box-filtered `value_noise(p)`: keep lattice coordinates exact via `Variant::Point`
            // (so `floor(p)` inside `value_noise` never fractionalises before hashing), and fade
            // both value and tangents smoothly to the analytic mean (0.0) as the widest per-axis
            // lattice footprint crosses Nyquist [0.35, 1.5] cells.
            let at = self.track.at;
            let dg =
                derive(self.m, self.cache, g, mask.clone(), self.n, self.target, Variant::Point)
                    .map_err(|e| e.through(at))?;
            let p_ty = self.m.functions[g.index()].params[0].ty;
            let mut new_args = vec![args[0].clone()];
            let mut w_sum: Option<ValueId> = None;
            for k in 0..self.n as usize {
                let (arg_k, tk_val) = match &args[0] {
                    Arg::Value(x) => {
                        let tv = self.t(*x, k);
                        (Arg::Value(self.tangent_or_zero(out, tv, p_ty)), tv)
                    }
                    Arg::Place(pl) => {
                        let tpl = self.place(pl, k)?;
                        let tv = self.emit(out, p_ty, Expr::Load(tpl.clone()));
                        (Arg::Place(tpl), Some(tv))
                    }
                };
                new_args.push(arg_k);
                if let Some(tk) = tk_val {
                    let size = self.builtin(out, Builtin::Abs, vec![tk], p_ty);
                    w_sum = Some(match w_sum {
                        None => size,
                        Some(s) => self.bin(out, BinOp::Add, s, size, p_ty),
                    });
                }
            }
            let dual = self.m.functions[dg.index()]
                .ret
                .ok_or_else(|| Error::internal("a missing result type"))?;
            let r = self.emit(out, dual, Expr::Call(dg, new_args));
            let ty = self.ty(v);
            let raw_v = self.emit(out, ty, Expr::Extract(r, 0));
            let atten = if let Some(w_vec) = w_sum {
                let wx = self.emit(out, ty, Expr::Extract(w_vec, 0));
                let wy = self.emit(out, ty, Expr::Extract(w_vec, 1));
                let wz = self.emit(out, ty, Expr::Extract(w_vec, 2));
                let wxy = self.builtin(out, Builtin::Max, vec![wx, wy], ty);
                let wmax = self.builtin(out, Builtin::Max, vec![wxy, wz], ty);
                let lo = self.fc(out, ty, 0.35);
                let hi = self.fc(out, ty, 1.5);
                let one = self.fc(out, ty, 1.0);
                let fade = self.builtin(out, Builtin::Smoothstep, vec![lo, hi, wmax], ty);
                self.bin(out, BinOp::Sub, one, fade, ty)
            } else {
                self.fc(out, ty, 1.0)
            };
            out.push(Stmt::Let(v, Expr::Binary(BinOp::Mul, raw_v, atten)));
            let mut ts = Vec::new();
            for k in 0..self.n as usize {
                let dt = self.emit(out, ty, Expr::Extract(r, 1 + k as u32));
                let adt = self.bin(out, BinOp::Mul, dt, atten, ty);
                ts.push(Some(adt));
            }
            self.tv.insert(v, ts);
            return Ok(());
        }
        if self.bounds
            && let (Some(v), true) = (v, returns)
            && self.m.functions[g.index()].known == Some(Known::Smin)
            && let [Arg::Value(a), Arg::Value(b), Arg::Value(k)] = args
            && mask[2] == 0
        {
            // `smin`'s derivative as the mix it is, `mix(da, db, w)` with `w` the weight on `b`
            // (`smin_weight`, in [0, 1]), whose interval is the hull of the two: written out,
            // `da + 2 m s (db - da)` counts `da` twice, and a fold of n smooth unions grows the
            // interval by about 1.5 each step.
            out.push(Stmt::Let(v, Expr::Call(g, args.to_vec())));
            let ty = self.ty(v);
            let d = self.bin(out, BinOp::Sub, *a, *b, ty);
            let ad = self.builtin(out, Builtin::Abs, vec![d], ty);
            let km = self.bin(out, BinOp::Sub, *k, ad, ty);
            let zero = self.fc(out, ty, 0.0);
            let m = self.builtin(out, Builtin::Max, vec![km, zero], ty);
            let tiny = self.fc(out, ty, 1.1754944e-38);
            let kk = self.builtin(out, Builtin::Max, vec![*k, tiny], ty);
            let half = self.fc(out, ty, 0.5);
            let s = self.bin(out, BinOp::Div, half, kk, ty);
            let w = self.bin(out, BinOp::Mul, m, s, ty);
            let w = self.builtin(out, Builtin::Min, vec![w, half], ty);
            let one = self.fc(out, ty, 1.0);
            let omw = self.bin(out, BinOp::Sub, one, w, ty);
            let bt = self.m.types.bool();
            let below = self.emit(out, bt, Expr::Binary(BinOp::Lt, *a, *b));
            let wb = self.emit(out, ty, Expr::Select { cond: below, if_true: w, if_false: omw });
            // Held to [0, 1] (it is, but an interval of it widens past its ends by its
            // rounding, and `mix`'s interval is the hull only for a weight inside them).
            let wb = self.builtin(out, Builtin::Max, vec![wb, zero], ty);
            let wb = self.builtin(out, Builtin::Min, vec![wb, one], ty);
            let mut ts = Vec::new();
            for dir in 0..self.n as usize {
                let (da, db) = (self.t(*a, dir), self.t(*b, dir));
                let da = self.tangent_or_zero(out, da, ty);
                let db = self.tangent_or_zero(out, db, ty);
                ts.push(Some(self.builtin(out, Builtin::Mix, vec![da, db, wb], ty)));
            }
            self.tv.insert(v, ts);
            return Ok(());
        }
        let at = self.track.at;
        let dg = derive(self.m, self.cache, g, mask.clone(), self.n, self.target, self.variant())
            .map_err(|e| e.through(at))?;
        let mut new_args = args.to_vec();
        for (i, a) in args.iter().enumerate() {
            if mask[i] == 0 {
                continue;
            }
            for k in 0..self.n as usize {
                new_args.push(match a {
                    Arg::Value(x) => {
                        let ty = self.m.functions[g.index()].params[i].ty;
                        Arg::Value(self.tangent_or_zero(out, self.t(*x, k), ty))
                    }
                    Arg::Place(p) => Arg::Place(self.place(p, k)?),
                });
            }
        }
        match (v, returns) {
            (Some(v), true) => {
                let dual = self.m.functions[dg.index()]
                    .ret
                    .ok_or_else(|| Error::internal("a missing result type"))?;
                let r = self.emit(out, dual, Expr::Call(dg, new_args));
                out.push(Stmt::Let(v, Expr::Extract(r, 0)));
                let ty = self.ty(v);
                let mut ts = Vec::new();
                for k in 0..self.n as usize {
                    ts.push(Some(self.emit(out, ty, Expr::Extract(r, 1 + k as u32))));
                }
                self.tv.insert(v, ts);
            }
            (v, _) => out.push(let_or_eval(v, Expr::Call(dg, new_args))),
        }
        Ok(())
    }

    /// The footprint of `a - b` (`b` none: of `a`): the sum over the directions of its tangent's
    /// size, its range over the footprint to first order; none where it has no tangent.
    fn footprint(
        &mut self,
        out: &mut Block,
        a: ValueId,
        b: Option<ValueId>,
        ty: TypeId,
    ) -> Option<ValueId> {
        let mut sum: Option<ValueId> = None;
        for k in 0..self.n as usize {
            let ta = self.t(a, k);
            let tb = b.and_then(|b| self.t(b, k));
            let t = match (ta, tb) {
                (None, None) => continue,
                (Some(x), None) => x,
                (None, Some(y)) => self.emit(out, ty, Expr::Unary(UnOp::Neg, y)),
                (Some(x), Some(y)) => self.bin(out, BinOp::Sub, x, y, ty),
            };
            let size = self.builtin(out, Builtin::Abs, vec![t], ty);
            sum = Some(match sum {
                None => size,
                Some(s) => self.bin(out, BinOp::Add, s, size, ty),
            });
        }
        sum
    }

    /// `fract(x)` box-filtered over a footprint `w` wide (`w` at least `NARROWEST`), from
    /// `f = fract(x)` so large `x` keeps its precision. Under a period wide, `f` corrected by the
    /// share of the footprint past each of `f`'s jumps (`h - f` below 0, `h - (1 - f)` past 1:
    /// differences of nearby values, exact), so a footprint crossing none is `f` exactly; wider,
    /// the difference of fract's integral (`u²/2` within a period, a half a period) at `f ± w/2`,
    /// over `w`.
    fn filtered_fract(&mut self, out: &mut Block, x: ValueId, w: ValueId, ty: TypeId) -> ValueId {
        let f = self.builtin(out, Builtin::Fract, vec![x], ty);
        let half = self.fc(out, ty, 0.5);
        let one = self.fc(out, ty, 1.0);
        let zero = self.zero(out, ty);
        let h = self.bin(out, BinOp::Mul, w, half, ty);
        let below = self.bin(out, BinOp::Sub, h, f, ty);
        let below = self.builtin(out, Builtin::Max, vec![below, zero], ty);
        let to_one = self.bin(out, BinOp::Sub, one, f, ty);
        let past = self.bin(out, BinOp::Sub, h, to_one, ty);
        let past = self.builtin(out, Builtin::Max, vec![past, zero], ty);
        let jumps = self.bin(out, BinOp::Sub, below, past, ty);
        let jumps = self.bin(out, BinOp::Div, jumps, w, ty);
        let narrow = self.bin(out, BinOp::Add, f, jumps, ty);
        let hi = self.bin(out, BinOp::Add, f, h, ty);
        let lo = self.bin(out, BinOp::Sub, f, h, ty);
        let ihi = self.fract_integral(out, hi, ty);
        let ilo = self.fract_integral(out, lo, ty);
        let d = self.bin(out, BinOp::Sub, ihi, ilo, ty);
        let wide = self.bin(out, BinOp::Div, d, w, ty);
        let is_wide = self.builtin(out, Builtin::Step, vec![one, w], ty);
        self.builtin(out, Builtin::Mix, vec![narrow, wide, is_wide], ty)
    }

    /// `∫₀ᵘ fract`: `floor(u) / 2 + fract(u)² / 2`.
    fn fract_integral(&mut self, out: &mut Block, u: ValueId, ty: TypeId) -> ValueId {
        let half = self.fc(out, ty, 0.5);
        let fl = self.builtin(out, Builtin::Floor, vec![u], ty);
        let fr = self.builtin(out, Builtin::Fract, vec![u], ty);
        let fr2 = self.bin(out, BinOp::Mul, fr, fr, ty);
        let s = self.bin(out, BinOp::Add, fl, fr2, ty);
        self.bin(out, BinOp::Mul, s, half, ty)
    }

    /// What box-filtering `|d|` over a footprint `w` wide adds to `|d|`: `max(w/2 - |d|, 0)² / w`
    /// (the filtered value is `(d² + w²/4) / w` within `w/2` of 0, `|d|` beyond). Computed from
    /// `|d|` and not from the integral's ends, so it's exactly 0 where the footprint misses the
    /// kink however large `d` is.
    fn kink(&mut self, out: &mut Block, d: ValueId, w: ValueId, ty: TypeId) -> ValueId {
        let half = self.fc(out, ty, 0.5);
        let zero = self.zero(out, ty);
        let h = self.bin(out, BinOp::Mul, w, half, ty);
        let a = self.builtin(out, Builtin::Abs, vec![d], ty);
        let inside = self.bin(out, BinOp::Sub, h, a, ty);
        let inside = self.builtin(out, Builtin::Max, vec![inside, zero], ty);
        let sq = self.bin(out, BinOp::Mul, inside, inside, ty);
        self.bin(out, BinOp::Div, sq, w, ty)
    }

    /// `step(0, x)` box-filtered over a footprint `w` wide: the share of it past 0.
    fn filtered_step(&mut self, out: &mut Block, x: ValueId, w: ValueId, ty: TypeId) -> ValueId {
        let half = self.fc(out, ty, 0.5);
        let r = self.bin(out, BinOp::Div, x, w, ty);
        let r = self.bin(out, BinOp::Add, r, half, ty);
        self.builtin(out, Builtin::Saturate, vec![r], ty)
    }

    /// A footprint at least `NARROWEST` wide.
    fn widened(&mut self, out: &mut Block, w: ValueId, ty: TypeId) -> ValueId {
        let narrowest = self.fc(out, ty, NARROWEST);
        self.builtin(out, Builtin::Max, vec![w, narrowest], ty)
    }

    /// `v = b(args)` in a filtering derivation: `b`'s value box-filtered over its operand's
    /// footprint (`footprint`), or its own value where the footprint is narrower than
    /// `NARROWEST` (a `mix` with a weight of 0 keeps it exactly). The kinked builtins are their
    /// own value and the corrections their kinks add (`kink`): `abs` `|x| + k(x)`, `min` and
    /// `max` `∓ k(a - b) / 2`, the clamps `+ (k(x - lo) - k(x - hi)) / 2` (each over its own
    /// difference's footprint). `floor` is `x - fract(x)`, `ceil` `x + fract(-x)`, `round`
    /// `floor(x + 1/2)`, `trunc` `floor(x) + 1 - step(0, x)` (they differ only at the
    /// integers), `sign` `2 step(0, x) - 1`.
    fn filtered_let(&mut self, v: ValueId, b: Builtin, args: &[ValueId], out: &mut Block) {
        use Builtin as B;
        let ty = self.ty(v);
        let own = self.emit(out, ty, Expr::Builtin(b, args.to_vec()));
        // The operand whose footprint matters: a step's edge from its point, min's and max's
        // difference, the clamps' the wider of their argument less each bound's, the rest's
        // argument.
        let (a, minus) = match b {
            B::Step => (args[1], Some(args[0])),
            B::Min | B::Max => (args[0], Some(args[1])),
            _ => (args[0], None),
        };
        let (w, bounds) = if b == B::Clamp {
            let wl = self.footprint(out, a, Some(args[1]), ty);
            let wh = self.footprint(out, a, Some(args[2]), ty);
            let w = match (wl, wh) {
                (Some(l), Some(h)) => Some(self.builtin(out, B::Max, vec![l, h], ty)),
                (l, h) => l.or(h),
            };
            (w, Some((wl, wh)))
        } else if b == B::Smoothstep {
            let wl = self.footprint(out, args[2], Some(args[0]), ty);
            let wh = self.footprint(out, args[2], Some(args[1]), ty);
            let w = match (wl, wh) {
                (Some(l), Some(h)) => Some(self.builtin(out, B::Max, vec![l, h], ty)),
                (l, h) => l.or(h),
            };
            (w, None)
        } else {
            (self.footprint(out, a, minus, ty), None)
        };
        let Some(w) = w else {
            out.push(Stmt::Let(v, Expr::Builtin(b, args.to_vec())));
            return;
        };
        let narrowest = self.fc(out, ty, NARROWEST);
        let used = self.builtin(out, B::Step, vec![narrowest, w], ty);
        let w = self.widened(out, w, ty);
        let one = self.fc(out, ty, 1.0);
        let half = self.fc(out, ty, 0.5);
        let x = args[0];
        let filtered = match b {
            B::Fract => self.filtered_fract(out, x, w, ty),
            B::Floor => {
                let f = self.filtered_fract(out, x, w, ty);
                self.bin(out, BinOp::Sub, x, f, ty)
            }
            B::Ceil => {
                let nx = self.emit(out, ty, Expr::Unary(UnOp::Neg, x));
                let f = self.filtered_fract(out, nx, w, ty);
                self.bin(out, BinOp::Add, x, f, ty)
            }
            B::Round => {
                let xh = self.bin(out, BinOp::Add, x, half, ty);
                let f = self.filtered_fract(out, xh, w, ty);
                self.bin(out, BinOp::Sub, xh, f, ty)
            }
            B::Trunc => {
                let f = self.filtered_fract(out, x, w, ty);
                let fl = self.bin(out, BinOp::Sub, x, f, ty);
                let st = self.filtered_step(out, x, w, ty);
                let up = self.bin(out, BinOp::Sub, one, st, ty);
                self.bin(out, BinOp::Add, fl, up, ty)
            }
            B::Step => {
                let d = self.bin(out, BinOp::Sub, args[1], args[0], ty);
                self.filtered_step(out, d, w, ty)
            }
            B::Sign => {
                let st = self.filtered_step(out, x, w, ty);
                let two = self.fc(out, ty, 2.0);
                let t = self.bin(out, BinOp::Mul, st, two, ty);
                self.bin(out, BinOp::Sub, t, one, ty)
            }
            B::Abs => {
                let k = self.kink(out, x, w, ty);
                self.bin(out, BinOp::Add, own, k, ty)
            }
            B::Min | B::Max => {
                let d = self.bin(out, BinOp::Sub, args[0], args[1], ty);
                let k = self.kink(out, d, w, ty);
                let k = self.bin(out, BinOp::Mul, k, half, ty);
                let op = if b == B::Min { BinOp::Sub } else { BinOp::Add };
                self.bin(out, op, own, k, ty)
            }
            B::Clamp | B::Saturate => {
                let (lo, hi) =
                    if b == B::Clamp { (args[1], args[2]) } else { (self.zero(out, ty), one) };
                // Each bound's kink over the footprint of `x` less it (a bound with no tangent
                // shares `x`'s).
                let (wl, wh) = bounds.unwrap_or_default();
                let wl = match wl {
                    Some(l) => self.widened(out, l, ty),
                    None => w,
                };
                let wh = match wh {
                    Some(h) => self.widened(out, h, ty),
                    None => w,
                };
                let xl = self.bin(out, BinOp::Sub, x, lo, ty);
                let xh = self.bin(out, BinOp::Sub, x, hi, ty);
                let kl = self.kink(out, xl, wl, ty);
                let kh = self.kink(out, xh, wh, ty);
                let k = self.bin(out, BinOp::Sub, kl, kh, ty);
                let k = self.bin(out, BinOp::Mul, k, half, ty);
                self.bin(out, BinOp::Add, own, k, ty)
            }
            B::Sin | B::Cos => {
                // Exact box integral over [x - w/2, x + w/2]: sin(x)*sinc(w/2), cos(x)*sinc(w/2).
                let h = self.bin(out, BinOp::Mul, w, half, ty);
                let sh = self.builtin(out, B::Sin, vec![h], ty);
                let sinc = self.bin(out, BinOp::Div, sh, h, ty);
                self.bin(out, BinOp::Mul, own, sinc, ty)
            }
            B::Smoothstep => {
                // Exact analytical box integral of smoothstep(e0, e1, x) over [x - w/2, x + w/2],
                // factored to avoid cancellation for small w and large |u|:
                //   ∫ S(t) dt = C(saturate(t)) + max(t - 1, 0),  C(c) = c³ - c⁴/2.
                let (e0, e1, xv) = (args[0], args[1], args[2]);
                let zero = self.zero(out, ty);
                let two = self.fc(out, ty, 2.0);
                let span = self.bin(out, BinOp::Sub, e1, e0, ty);
                let aspan = self.builtin(out, B::Abs, vec![span], ty);
                let safe_aspan = self.widened(out, aspan, ty);
                let pos = self.builtin(out, B::Step, vec![zero, span], ty);
                let sgn2 = self.bin(out, BinOp::Mul, pos, two, ty);
                let sgn = self.bin(out, BinOp::Sub, sgn2, one, ty);
                let safe_span = self.bin(out, BinOp::Mul, sgn, safe_aspan, ty);
                let xe = self.bin(out, BinOp::Sub, xv, e0, ty);
                let u = self.bin(out, BinOp::Div, xe, safe_span, ty);
                let wu = self.bin(out, BinOp::Div, w, safe_aspan, ty);
                let wu = self.widened(out, wu, ty);
                let hw = self.bin(out, BinOp::Mul, wu, half, ty);
                let ua = self.bin(out, BinOp::Sub, u, hw, ty);
                let ub = self.bin(out, BinOp::Add, u, hw, ty);
                let ca = self.builtin(out, B::Saturate, vec![ua], ty);
                let cb = self.builtin(out, B::Saturate, vec![ub], ty);
                let dc = self.bin(out, BinOp::Sub, cb, ca, ty);
                let frac = self.bin(out, BinOp::Div, dc, wu, ty);
                let ca2 = self.bin(out, BinOp::Mul, ca, ca, ty);
                let cb2 = self.bin(out, BinOp::Mul, cb, cb, ty);
                let cab = self.bin(out, BinOp::Mul, ca, cb, ty);
                let sum_sq = self.bin(out, BinOp::Add, ca2, cb2, ty);
                let q1 = self.bin(out, BinOp::Add, sum_sq, cab, ty);
                let sum_c = self.bin(out, BinOp::Add, ca, cb, ty);
                let q2 = self.bin(out, BinOp::Mul, sum_c, sum_sq, ty);
                let q2 = self.bin(out, BinOp::Mul, q2, half, ty);
                let poly = self.bin(out, BinOp::Sub, q1, q2, ty);
                let curved = self.bin(out, BinOp::Mul, frac, poly, ty);
                let um1 = self.bin(out, BinOp::Sub, u, one, ty);
                let ramp = self.filtered_step(out, um1, wu, ty);
                self.bin(out, BinOp::Add, curved, ramp, ty)
            }
            _ => own,
        };
        out.push(Stmt::Let(v, Expr::Builtin(B::Mix, vec![own, filtered, used])));
    }

    /// The `k`th tangent of `v = e`.
    fn tangent(
        &mut self,
        v: ValueId,
        e: &Expr,
        k: usize,
        out: &mut Block,
    ) -> Result<Option<ValueId>> {
        let ty = self.ty(v);
        if !self.m.types.has_float(ty) {
            return Ok(None);
        }
        let t = |s: &Self, x: &ValueId| s.t(*x, k);
        Ok(match e {
            Expr::Const(_)
            | Expr::Zero(_)
            | Expr::EntryInput(_)
            | Expr::Bitcast(..)
            | Expr::ArrayLength(_) => None,
            Expr::Param(i) => {
                self.tparam[*i as usize].get(k).copied().map(|p| self.emit(out, ty, Expr::Param(p)))
            }
            Expr::Load(p) => {
                if self.act.place(self.m, self.f, p) && !matches!(p.root, PlaceRoot::Resource(_)) {
                    let tp = self.place(p, k)?;
                    Some(self.emit(out, ty, Expr::Load(tp)))
                } else {
                    None
                }
            }
            Expr::Unary(UnOp::Neg, x) => {
                t(self, x).map(|dx| self.emit(out, ty, Expr::Unary(UnOp::Neg, dx)))
            }
            Expr::Unary(UnOp::Not, _) => None,
            Expr::Binary(op, a, b) => {
                let (da, db) = (t(self, a), t(self, b));
                match op {
                    BinOp::Add => self.add(out, da, db, ty),
                    BinOp::Sub => self.sub(out, da, db, ty),
                    BinOp::Mul if self.track.same_value(*a, *b) && !self.is_matrix(*a) => {
                        // d(a²) = 2 a da (a matrix product doesn't commute: dM M + M dM)
                        da.map(|da| {
                            let two = self.fc(out, ty, 2.0);
                            let t = self.bin(out, BinOp::Mul, *a, two, ty);
                            self.scale(out, da, t, ty)
                        })
                    }
                    BinOp::Mul => {
                        let x = self.mul(out, da, *b, ty);
                        // A matrix product keeps its order: `a db`, also for a vector times a
                        // matrix (`v dM`, not `dM v`).
                        let y = db.map(|db| {
                            if self.is_matrix(*a) || self.is_matrix(db) {
                                self.bin(out, BinOp::Mul, *a, db, ty)
                            } else {
                                self.scale(out, db, *a, ty)
                            }
                        });
                        self.add(out, x, y, ty)
                    }
                    BinOp::Div => {
                        // (da - v db) / b
                        let vdb = db.map(|db| self.scale(out, db, v, ty));
                        let num = self.sub(out, da, vdb, ty);
                        num.map(|n| self.div(out, n, *b, ty))
                    }
                    BinOp::Rem => {
                        // a - b trunc(a / b): da - db trunc(a / b)
                        let q = self.bin(out, BinOp::Div, *a, *b, ty);
                        let q = self.builtin(out, Builtin::Trunc, vec![q], ty);
                        let dbq = db.map(|db| self.scale(out, db, q, ty));
                        self.sub(out, da, dbq, ty)
                    }
                    _ => None,
                }
            }
            Expr::Builtin(b, args) => self.builtin_tangent(v, *b, args, k, ty, out)?,
            Expr::Construct(cty, parts) => {
                let ts: Vec<Option<ValueId>> = parts.iter().map(|p| t(self, p)).collect();
                if ts.iter().all(Option::is_none) {
                    None
                } else {
                    let mut comps = Vec::new();
                    for (p, tp) in parts.iter().zip(ts) {
                        let pty = self.ty(*p);
                        comps.push(self.tangent_or_zero(out, tp, pty));
                    }
                    Some(self.emit(out, *cty, Expr::Construct(*cty, comps)))
                }
            }
            // The payload's tangent, in the same variant.
            Expr::Variant(et, k, payload) => payload
                .and_then(|p| t(self, &p))
                .map(|dp| self.emit(out, *et, Expr::Variant(*et, *k, Some(dp)))),
            Expr::Extract(x, i) => t(self, x).map(|dx| self.emit(out, ty, Expr::Extract(dx, *i))),
            Expr::ExtractDyn(x, i) => {
                t(self, x).map(|dx| self.emit(out, ty, Expr::ExtractDyn(dx, *i)))
            }
            Expr::Splat(x, n) => t(self, x).map(|dx| self.emit(out, ty, Expr::Splat(dx, *n))),
            Expr::Swizzle(x, c) => {
                t(self, x).map(|dx| self.emit(out, ty, Expr::Swizzle(dx, c.clone())))
            }
            Expr::Convert(x, s) => {
                if s.is_float()
                    && self.m.types.element_scalar(self.ty(*x)).is_some_and(|s| s.is_float())
                {
                    t(self, x).map(|dx| self.emit(out, ty, Expr::Convert(dx, *s)))
                } else {
                    None
                }
            }
            Expr::Select { cond, if_true, if_false } => {
                let (dt, df) = (t(self, if_true), t(self, if_false));
                if dt.is_none() && df.is_none() {
                    None
                } else {
                    let dt = self.tangent_or_zero(out, dt, ty);
                    let df = self.tangent_or_zero(out, df, ty);
                    Some(self.emit(
                        out,
                        ty,
                        Expr::Select { cond: *cond, if_true: dt, if_false: df },
                    ))
                }
            }
            Expr::Call(..) => unreachable!("calls are handled in `call`"),
            Expr::Addr(p) => {
                if self.act.root(&p.root) || self.act.index_active(p) {
                    let tp = self.place(p, k)?;
                    Some(self.emit(out, ty, Expr::Addr(tp)))
                } else {
                    None
                }
            }
            Expr::Run(_) => {
                return Err(Error::not_derivable(
                    "can't differentiate through a run (`[T]`) of values that depend on the input yet: pass the array itself (`[f32; N]`)",
                ));
            }
            Expr::Texture(..) | Expr::TextureStore(..) => {
                return Err(Error::not_derivable(
                    "can't differentiate through a texture read: its texels are data, not code",
                ));
            }
            Expr::Atomic(..) | Expr::Barrier | Expr::Subgroup(..) | Expr::Discard => {
                return Err(Error::not_derivable(
                    "can't differentiate through an atomic or a barrier: derived code is pure",
                ));
            }
            Expr::Host(..) => {
                return Err(Error::not_derivable(
                    "can't differentiate GPU work (a buffer, a dispatch or a draw)",
                ));
            }
            Expr::Mem(..) => {
                return Err(Error::not_derivable(
                    "can't differentiate code that allocates or reads raw memory",
                ));
            }
        })
    }

    fn builtin_tangent(
        &mut self,
        v: ValueId,
        b: Builtin,
        args: &[ValueId],
        k: usize,
        ty: TypeId,
        out: &mut Block,
    ) -> Result<Option<ValueId>> {
        use Builtin as B;
        let d: Vec<Option<ValueId>> = args.iter().map(|a| self.t(*a, k)).collect();
        let a0 = args[0];
        let a0ty = self.ty(a0);
        let ln2 = std::f64::consts::LN_2;
        Ok(match b {
            B::Floor | B::Ceil | B::Round | B::Trunc | B::Sign | B::Step | B::AllEqual => None,
            // Integers: no tangent.
            B::CountOnes | B::LeadingZeros | B::TrailingZeros => None,
            B::Dpdx | B::Dpdy | B::Fwidth => {
                return Err(Error::not_derivable(
                    "derivatives of screen-space derivatives aren't supported",
                ));
            }
            B::Fract => d[0],
            B::Sqrt => {
                // A length written out is a scalar's root; a vector's is a root per component.
                if self.bounds
                    && d[0].is_some()
                    && self.m.types.as_scalar(ty).is_some()
                    && let Some((terms, factor)) = self.sum_of_squares(a0)
                    && let Some(t) = self.root_of_squares(&terms, factor, k, ty, out)
                {
                    Some(t)
                } else {
                    d[0].map(|da| {
                        let two = self.fc(out, ty, 2.0);
                        let den = self.bin(out, BinOp::Mul, v, two, ty);
                        self.div_from_zero(out, da, den, ty)
                    })
                }
            }
            B::InverseSqrt => d[0].map(|da| {
                let h = self.fc(out, ty, -0.5);
                let r2 = self.bin(out, BinOp::Mul, v, v, ty);
                let r3 = self.bin(out, BinOp::Mul, r2, v, ty);
                let f = self.bin(out, BinOp::Mul, r3, h, ty);
                self.scale(out, da, f, ty)
            }),
            B::Sin => d[0].map(|da| {
                let c = self.builtin(out, B::Cos, vec![a0], ty);
                self.scale(out, da, c, ty)
            }),
            B::Cos => d[0].map(|da| {
                let s = self.builtin(out, B::Sin, vec![a0], ty);
                let ns = self.emit(out, ty, Expr::Unary(UnOp::Neg, s));
                self.scale(out, da, ns, ty)
            }),
            B::Tan => d[0].map(|da| {
                let one = self.fc(out, ty, 1.0);
                let r2 = self.bin(out, BinOp::Mul, v, v, ty);
                let f = self.bin(out, BinOp::Add, r2, one, ty);
                self.scale(out, da, f, ty)
            }),
            B::Asin | B::Acos => d[0].map(|da| {
                let one = self.fc(out, ty, 1.0);
                let a2 = self.bin(out, BinOp::Mul, a0, a0, ty);
                let s = self.bin(out, BinOp::Sub, one, a2, ty);
                let r = self.builtin(out, B::Sqrt, vec![s], ty);
                let q = self.div(out, da, r, ty);
                if b == B::Acos { self.emit(out, ty, Expr::Unary(UnOp::Neg, q)) } else { q }
            }),
            B::Atan => d[0].map(|da| {
                let one = self.fc(out, ty, 1.0);
                let a2 = self.bin(out, BinOp::Mul, a0, a0, ty);
                let s = self.bin(out, BinOp::Add, a2, one, ty);
                self.div(out, da, s, ty)
            }),
            B::Atan2 => {
                // (x dy - y dx) / (x² + y²), with y = args[0], x = args[1].
                let (y, x) = (args[0], args[1]);
                let xdy = d[0].map(|dy| self.scale(out, dy, x, ty));
                let ydx = d[1].map(|dx| self.scale(out, dx, y, ty));
                let num = self.sub(out, xdy, ydx, ty);
                num.map(|n| {
                    let x2 = self.bin(out, BinOp::Mul, x, x, ty);
                    let y2 = self.bin(out, BinOp::Mul, y, y, ty);
                    let den = self.bin(out, BinOp::Add, x2, y2, ty);
                    self.div(out, n, den, ty)
                })
            }
            B::Exp => d[0].map(|da| self.scale(out, da, v, ty)),
            B::Sinh => d[0].map(|da| {
                let c = self.builtin(out, B::Cosh, vec![a0], ty);
                self.scale(out, da, c, ty)
            }),
            B::Cosh => d[0].map(|da| {
                let s = self.builtin(out, B::Sinh, vec![a0], ty);
                self.scale(out, da, s, ty)
            }),
            // 1 - tanh², from the value: it stays in [0, 1] where cosh² would overflow.
            B::Tanh => d[0].map(|da| {
                let one = self.fc(out, ty, 1.0);
                let r2 = self.bin(out, BinOp::Mul, v, v, ty);
                let f = self.bin(out, BinOp::Sub, one, r2, ty);
                self.scale(out, da, f, ty)
            }),
            B::Exp2 => d[0].map(|da| {
                let c = self.fc(out, ty, ln2);
                let f = self.bin(out, BinOp::Mul, v, c, ty);
                self.scale(out, da, f, ty)
            }),
            B::Log => d[0].map(|da| self.div(out, da, a0, ty)),
            B::Log2 => d[0].map(|da| {
                let c = self.fc(out, ty, ln2);
                let den = self.bin(out, BinOp::Mul, a0, c, ty);
                self.div(out, da, den, ty)
            }),
            B::Pow => {
                // b a^(b-1) da + a^b log(a) db. Each factor is made finite first: where a is 0
                // and doesn't move (da = 0, as in `pow(max(x, 0), 0.5)`), the term is 0, not
                // ∞ × 0.
                let (a, bb) = (args[0], args[1]);
                let big = self.fc(out, ty, f32::MAX as f64);
                let small = self.fc(out, ty, -f32::MAX as f64);
                let x = d[0].map(|da| {
                    let one = self.fc(out, ty, 1.0);
                    let bm1 = self.bin(out, BinOp::Sub, bb, one, ty);
                    let p = self.builtin(out, B::Pow, vec![a, bm1], ty);
                    let f = self.bin(out, BinOp::Mul, bb, p, ty);
                    let f = self.builtin(out, B::Clamp, vec![f, small, big], ty);
                    self.scale(out, da, f, ty)
                });
                let y = d[1].map(|db| {
                    let l = self.builtin(out, B::Log, vec![a], ty);
                    let l = self.builtin(out, B::Clamp, vec![l, small, big], ty);
                    let f = self.bin(out, BinOp::Mul, l, v, ty);
                    self.scale(out, db, f, ty)
                });
                self.add(out, x, y, ty)
            }
            B::Abs => d[0].map(|da| {
                let s = self.builtin(out, B::Sign, vec![a0], ty);
                self.scale(out, da, s, ty)
            }),
            // The taken argument's tangent; where the arguments are equal, the average of
            // both. That's the symmetric choice (as `abs`'s, zero at zero, is), and the
            // derivative of a smooth function written with kinks that cancel: std's `smin`,
            // `min(a, b) - f(|a - b|)`, has 1/2 in each on its seam.
            B::Min | B::Max if self.m.types.as_scalar(ty).is_some() => {
                if d[0].is_none() && d[1].is_none() {
                    return Ok(None);
                }
                let (a, bb) = (args[0], args[1]);
                let bt = self.m.types.bool();
                let op = if b == B::Min { BinOp::Lt } else { BinOp::Gt };
                let first = self.emit(out, bt, Expr::Binary(op, a, bb));
                let tie = self.emit(out, bt, Expr::Binary(BinOp::Eq, a, bb));
                let half = self.fc(out, ty, 0.5);
                if self.target == Target::Gpu {
                    // A weight w (1, 1/2 or 0), shared by the directions: one `mix` each. A
                    // `select` and an average each measured over the grazer's 1.25× (WGSL may
                    // assume the tangent not taken is finite, which `mix` needs).
                    let one = self.fc(out, ty, 1.0);
                    let zero = self.fc(out, ty, 0.0);
                    let w = self.emit(
                        out,
                        ty,
                        Expr::Select { cond: first, if_true: one, if_false: zero },
                    );
                    let w =
                        self.emit(out, ty, Expr::Select { cond: tie, if_true: half, if_false: w });
                    return Ok(Some(match (d[0], d[1]) {
                        (Some(x), Some(y)) => self.builtin(out, B::Mix, vec![y, x, w], ty),
                        (Some(x), None) => self.scale(out, x, w, ty),
                        (None, y) => {
                            let y = self.tangent_or_zero(out, y, ty);
                            let nw = self.bin(out, BinOp::Sub, one, w, ty);
                            self.scale(out, y, nw, ty)
                        }
                    }));
                }
                // On the CPU, selects: a tangent not taken may be infinite (a square root's at 0).
                let x = self.tangent_or_zero(out, d[0], ty);
                let y = self.tangent_or_zero(out, d[1], ty);
                let pick =
                    self.emit(out, ty, Expr::Select { cond: first, if_true: x, if_false: y });
                let sum = self.add(out, d[0], d[1], ty).unwrap_or(x);
                let avg = self.bin(out, BinOp::Mul, sum, half, ty);
                Some(self.emit(out, ty, Expr::Select { cond: tie, if_true: avg, if_false: pick }))
            }
            B::Min | B::Max => {
                // m is 1 where the first argument is taken, 0 where the second is, and 1/2 on a
                // tie (as for a scalar): (step(a, b) - step(b, a) + 1) / 2 for `min`.
                let (a, bb) = (args[0], args[1]);
                let (p, q) = if b == B::Min { (a, bb) } else { (bb, a) };
                let s = self.builtin(out, B::Step, vec![p, q], ty);
                let t = self.builtin(out, B::Step, vec![q, p], ty);
                let diff = self.bin(out, BinOp::Sub, s, t, ty);
                let half = self.fc(out, ty, 0.5);
                let h = self.bin(out, BinOp::Mul, diff, half, ty);
                let m = self.bin(out, BinOp::Add, h, half, ty);
                let one = self.fc(out, ty, 1.0);
                let not_m = self.bin(out, BinOp::Sub, one, m, ty);
                let x = self.mul(out, d[0], m, ty);
                let y = self.mul(out, d[1], not_m, ty);
                self.add(out, x, y, ty)
            }
            B::Clamp => {
                // min(max(x, lo), hi), on both targets: the max takes x or lo, then the min takes
                // that or hi. (With lo > hi, that's hi everywhere.)
                let (x, lo, hi) = (args[0], args[1], args[2]);
                let above_lo = self.builtin(out, B::Step, vec![lo, x], ty);
                let m = self.builtin(out, B::Max, vec![x, lo], ty);
                let below_hi = self.builtin(out, B::Step, vec![m, hi], ty);
                let inside = self.bin(out, BinOp::Mul, above_lo, below_hi, ty);
                let one = self.fc(out, ty, 1.0);
                let not_lo = self.bin(out, BinOp::Sub, one, above_lo, ty);
                let lo_taken = self.bin(out, BinOp::Mul, not_lo, below_hi, ty);
                let not_hi = self.bin(out, BinOp::Sub, one, below_hi, ty);
                let t1 = self.mul(out, d[0], inside, ty);
                let t2 = self.mul(out, d[1], lo_taken, ty);
                let t3 = self.mul(out, d[2], not_hi, ty);
                let s = self.add(out, t1, t2, ty);
                self.add(out, s, t3, ty)
            }
            B::Saturate => d[0].map(|dx| {
                let zero = self.zero(out, ty);
                let one = self.fc(out, ty, 1.0);
                let onev = self.coerce(out, one, ty);
                let a = self.builtin(out, B::Step, vec![zero, a0], ty);
                let bb = self.builtin(out, B::Step, vec![a0, onev], ty);
                let inside = self.bin(out, BinOp::Mul, a, bb, ty);
                self.scale(out, dx, inside, ty)
            }),
            B::Mix => {
                // da (1 - t) + db t + (b - a) dt
                let (a, bb, tt) = (args[0], args[1], args[2]);
                let ttype = self.ty(tt);
                let one = self.fc(out, ttype, 1.0);
                let omt = self.bin(out, BinOp::Sub, one, tt, ttype);
                let x = self.mul(out, d[0], omt, ty);
                let y = self.mul(out, d[1], tt, ty);
                let z = d[2].map(|dt| {
                    let ba = self.bin(out, BinOp::Sub, bb, a, ty);
                    self.scale(out, dt, ba, ty)
                });
                let s = self.add(out, x, y, ty);
                self.add(out, s, z, ty)
            }
            B::Smoothstep => {
                // u = (x - e0) / w, t = clamp(u, 0, 1): dr = 6 t (1 - t) du, where
                // du = (dx - de0 - u (de1 - de0)) / w. (6t(1-t) is 0 where the clamp acts.)
                let (e0, e1, x) = (args[0], args[1], args[2]);
                let w = self.bin(out, BinOp::Sub, e1, e0, ty);
                let xe = self.bin(out, BinOp::Sub, x, e0, ty);
                let u = self.bin(out, BinOp::Div, xe, w, ty);
                let de = self.sub(out, d[1], d[0], ty);
                let ude = de.map(|de| self.scale(out, de, u, ty));
                let n1 = self.sub(out, d[2], d[0], ty);
                let num = self.sub(out, n1, ude, ty);
                num.map(|num| {
                    let du = self.div(out, num, w, ty);
                    let zero = self.zero(out, ty);
                    let one = self.fc(out, ty, 1.0);
                    let onev = self.coerce(out, one, ty);
                    let t = self.builtin(out, B::Clamp, vec![u, zero, onev], ty);
                    let omt = self.bin(out, BinOp::Sub, onev, t, ty);
                    let six = self.fc(out, ty, 6.0);
                    let tt = self.bin(out, BinOp::Mul, t, omt, ty);
                    let f = self.bin(out, BinOp::Mul, tt, six, ty);
                    self.scale(out, du, f, ty)
                })
            }
            B::Length => d[0].map(|dx| {
                if self.m.types.is_vector(a0ty) {
                    // dot(normalize(x), dx), and 0 where |x| is 0. Written with `normalize` so
                    // that its interval is tight where x has one nonzero component (a box's
                    // face): there `x / |x|` is ±1, which dividing intervals can't see. The
                    // zero vector never reaches `normalize`, which would make a NaN.
                    let zero = self.fc(out, ty, 0.0);
                    let b = self.m.types.bool();
                    let pos = self.emit(out, b, Expr::Binary(BinOp::Gt, v, zero));
                    let n = match self.m.types.get(a0ty) {
                        TypeDef::Vector(_, n) => *n,
                        _ => 3,
                    };
                    let one = self.fc(out, ty, 1.0);
                    let mut axis = vec![one];
                    axis.extend((1..n).map(|_| zero));
                    let axis = self.emit(out, a0ty, Expr::Construct(a0ty, axis));
                    let safe = self.emit(
                        out,
                        a0ty,
                        Expr::Select { cond: pos, if_true: a0, if_false: axis },
                    );
                    let unit = self.builtin(out, B::Normalize, vec![safe], a0ty);
                    let none = self.emit(out, a0ty, Expr::Zero(a0ty));
                    let dir = self.emit(
                        out,
                        a0ty,
                        Expr::Select { cond: pos, if_true: unit, if_false: none },
                    );
                    self.builtin(out, B::Dot, vec![dir, dx], ty)
                } else {
                    let s = self.builtin(out, B::Sign, vec![a0], ty);
                    self.scale(out, dx, s, ty)
                }
            }),
            B::Distance => {
                let dd = self.sub(out, d[0], d[1], a0ty);
                dd.map(|dd| {
                    let diff = self.bin(out, BinOp::Sub, args[0], args[1], a0ty);
                    if self.m.types.is_vector(a0ty) {
                        let dot = self.builtin(out, B::Dot, vec![diff, dd], ty);
                        self.div_from_zero(out, dot, v, ty)
                    } else {
                        let s = self.builtin(out, B::Sign, vec![diff], ty);
                        self.scale(out, dd, s, ty)
                    }
                })
            }
            B::Dot if self.track.same_value(args[0], args[1]) => d[0].map(|da| {
                // d(v·v) = 2 v·dv
                let dot = self.builtin(out, B::Dot, vec![a0, da], ty);
                let two = self.fc(out, ty, 2.0);
                self.scale(out, dot, two, ty)
            }),
            B::Dot => {
                let x = d[0].map(|da| self.builtin(out, B::Dot, vec![da, args[1]], ty));
                let y = d[1].map(|db| self.builtin(out, B::Dot, vec![args[0], db], ty));
                self.add(out, x, y, ty)
            }
            B::Cross => {
                let x = d[0].map(|da| self.builtin(out, B::Cross, vec![da, args[1]], ty));
                let y = d[1].map(|db| self.builtin(out, B::Cross, vec![args[0], db], ty));
                self.add(out, x, y, ty)
            }
            B::Normalize => d[0].map(|dx| {
                // (dx - n dot(n, dx)) / |x|
                let scalar = self.m.types.scalar(self.m.types.element_scalar(a0ty).unwrap());
                let len = self.builtin(out, B::Length, vec![a0], scalar);
                let ndx = self.builtin(out, B::Dot, vec![v, dx], scalar);
                let proj = self.bin(out, BinOp::Mul, v, ndx, ty);
                let num = self.bin(out, BinOp::Sub, dx, proj, ty);
                self.div(out, num, len, ty)
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_vector_normalization_has_a_valid_derivative() {
        let mut m = Module::default();
        let ty = m.types.vector_of(Scalar::F64, 2);
        let mut f = Function::new("normalize_double", vec![Param::value("x", ty)], Some(ty));
        let mut body = Vec::new();
        let x = f.let_(&mut body, ty, Expr::Param(0));
        let normalized = f.let_(&mut body, ty, Expr::Builtin(Builtin::Normalize, vec![x]));
        body.push(Stmt::Return(Some(normalized)));
        f.body = body;
        let id = m.add_function(f);
        verify(&m).expect("the source is valid IR");
        derive(&mut m, &mut DeriveCache::default(), id, vec![1], 1, Target::Cpu, Variant::Point)
            .expect("normalization is differentiable");
        verify(&m).expect("the derivative keeps the vector's component type");
    }

    /// `sqrt(x * x)` of a vector is a root per component, not a length: the bounded twin keeps
    /// `sqrt`'s own rule for it, and its code is valid.
    #[test]
    fn a_vectors_root_of_its_squares_has_a_valid_bounded_derivative() {
        let mut m = Module::default();
        let ty = m.types.vector_of(Scalar::F32, 3);
        let mut f = Function::new("root_of_squares", vec![Param::value("x", ty)], Some(ty));
        let mut body = Vec::new();
        let x = f.let_(&mut body, ty, Expr::Param(0));
        let xx = f.let_(&mut body, ty, Expr::Binary(BinOp::Mul, x, x));
        let r = f.let_(&mut body, ty, Expr::Builtin(Builtin::Sqrt, vec![xx]));
        body.push(Stmt::Return(Some(r)));
        f.body = body;
        let id = m.add_function(f);
        verify(&m).expect("the source is valid IR");
        derive(&mut m, &mut DeriveCache::default(), id, vec![1], 1, Target::Cpu, Variant::Bounded)
            .expect("a vector's root is differentiable");
        verify(&m).expect("the bounded derivative of a vector's root is valid IR");
    }
}
