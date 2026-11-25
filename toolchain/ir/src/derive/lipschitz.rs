//! Hybrid Clarke-subdifferential + interval-Jacobian Lipschitz derivation (`std::derive::lipschitz`).
//!
//! Computes a sound, IEEE-754/WGSL-rounded-up upper bound `L >= sup_{x ∈ over} ||∇f(x)||₂` for
//! any scalar function `f: X -> f32` over a box `over` (`X ∈ {f32, vec2, vec3, vec4}`).
//!
//! Each active float value `v` carries:
//! - its value interval `[lo, hi]` (same IR type as `v`),
//! - its component-wise Jacobian interval `[j_lo[i], j_hi[i]]` (`i = 0..n`, column `i` of `J_v`),
//! - its scalar Clarke-Lipschitz upper bound `lip: f32 >= ||J_v(x)||₂` (operator 2-norm).
//!
//! Why both `J` and `lip`:
//! - Tracking `J` preserves exact linear cancellation (`pa - u * dot(pa, u)`, `x - x`, orthogonal
//!   rotations `R * p` where Gershgorin on `J Jᵀ = I` gives exact `1.0`).
//! - Tracking scalar `lip` eliminates crease blowup across the continuous piecewise builtins
//!   (`abs`, `min`, `max`, `clamp`, `length`): `max(lip_a, lip_b)` stays `1.0` across 90° box
//!   edges and union creases where component-wise `||hull(∇a, ∇b)||₂` jumps to `√2` or `√3`.
//!
//! What jumps has no bound. A branch on the input (`if`, a select) whose condition the box
//! doesn't settle may switch between values that differ on the switching surface, and a step
//! function (`floor`, `ceil`, `round`, `trunc`, `sign`, `step`, `fract`) whose box holds a step
//! jumps there: either is `BIG`, its Jacobian unbounded too (so no `tighten` takes it back). A box
//! that settles the condition, or holds no step, keeps the bound of what it takes. (`max(t, f)`
//! across a switch holds only for a function continuous there, which nothing here can check:
//! `if x > 0 { x } else { x + 1 }` is 1 each side and unbounded across.) A branch on a condition
//! `std::field::meets(at)` made is where the author states the two sides agree (an exact
//! distance's regions): across it the bound is the steeper side's.

use super::*;
use std::collections::{HashMap, HashSet};

type R<T> = Result<T>;

const EPS: f64 = 1.0 / 8_388_608.0;
const BIG: f64 = 1.0e30;
const CAP: f64 = 1.0e15;
const NOISE_GRAD_MAX: f64 = 3.75;
const NOISE_HESS_DIAG: f64 = 11.55;
const NOISE_HESS_MIX: f64 = 14.07;
const NOISE_ROUNDING: f64 = 64.0 * EPS;

/// Active float value state: value bounds `[lo, hi]`, Jacobian column intervals `[j_lo, j_hi]`
/// (`n` columns, each of the value's IR type), and scalar operator-2-norm bound `lip: f32`.
#[derive(Clone, Debug)]
struct LVal {
    lo: ValueId,
    hi: ValueId,
    j_lo: Vec<ValueId>,
    j_hi: Vec<ValueId>,
    lip: ValueId,
}

type Sure = (ValueId, ValueId);

enum LState {
    Float(LVal),
    Bool(Sure),
}

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

fn tiny(target: Target) -> f64 {
    match target {
        Target::Cpu => 1.5e-45,
        Target::Gpu => 1.2e-38,
    }
}

/// Entry point for `std::derive::lipschitz(f, over) -> f32`.
pub(super) fn lipschitz(
    m: &mut Module,
    cache: &mut DeriveCache,
    f: FuncId,
    ncap: u32,
    target: Target,
    box_ty: TypeId,
) -> R<FuncId> {
    let func = m.functions[f.index()].clone();
    let xi = ncap as usize;
    let x_ty = func.params[xi].ty;
    let f32_ty = m.types.f32();
    let n = match m.types.get(x_ty) {
        TypeDef::Scalar(Scalar::F32) => 1usize,
        &TypeDef::Vector(Scalar::F32, k) => k as usize,
        _ => return Err(Error::not_derivable("a Lipschitz bound of something that isn't a float")),
    };
    if func.ret != Some(f32_ty) {
        return Err(Error::not_derivable("a Lipschitz bound of a non-f32 result"));
    }
    let mut params = func.params[..xi].to_vec();
    params.push(Param::value("over", box_ty));
    let mut g = Function::new(format!("{}_lipschitz", func.name), params, Some(f32_ty));
    let mut body = Vec::new();
    let mut mask = vec![0; func.params.len()];
    mask[xi] = all_bits(&m.types, x_ty);
    let over = g.let_(&mut body, box_ty, Expr::Param(xi as u32));
    if returns_active(m, cache, f, mask.clone(), Mode::Interval) == 0 {
        let z = g.let_(&mut body, f32_ty, Expr::Const(Const::F32(0.0)));
        body.push(Stmt::Return(Some(z)));
        g.body = body;
        return Ok(m.add_function(g));
    }
    let derived = match derive(m, cache, f, mask, n, target) {
        Ok(d) => d,
        Err(Error::NotDerivable(..) | Error::ActiveLoopExit(..)) => {
            // Fallback: return `BIG` when the function cannot be Clarke-Lipschitz derived.
            let big = g.let_(&mut body, f32_ty, Expr::Const(Const::F32(BIG as f32)));
            body.push(Stmt::Return(Some(big)));
            g.body = body;
            return Ok(m.add_function(g));
        }
        Err(e) => return Err(e),
    };
    let d_params = m.functions[derived.index()].params.clone();
    let mut args = Vec::new();
    for i in 0..xi {
        if d_params[i].by_ref {
            args.push(Arg::Place(Place::root(PlaceRoot::Param(i as u32))));
        } else {
            let v = g.let_(&mut body, func.params[i].ty, Expr::Param(i as u32));
            args.push(Arg::Value(v));
        }
    }
    let lo = g.let_(&mut body, x_ty, Expr::Extract(over, 0));
    let hi = g.let_(&mut body, x_ty, Expr::Extract(over, 1));
    let pass = |g: &mut Function, body: &mut Block, v: ValueId, ty: TypeId, by_ref: bool| -> Arg {
        if !by_ref {
            return Arg::Value(v);
        }
        let l = g.new_local("arg", ty);
        body.push(Stmt::Store(Place::local(l), v));
        Arg::Place(Place::local(l))
    };
    args.push(pass(&mut g, &mut body, lo, x_ty, d_params[xi].by_ref));
    // Extra params: hi, j_lo[0..n], j_hi[0..n], lip.
    let mut extra_idx = xi + 1;
    args.push(pass(&mut g, &mut body, hi, x_ty, d_params[extra_idx].by_ref));
    extra_idx += 1;
    let zero = g.let_(&mut body, f32_ty, Expr::Const(Const::F32(0.0)));
    let one = g.let_(&mut body, f32_ty, Expr::Const(Const::F32(1.0)));
    let mut basis = Vec::with_capacity(n);
    for i in 0..n {
        let e_i = if n == 1 {
            one
        } else {
            let comps = (0..n).map(|j| if j == i { one } else { zero }).collect();
            g.let_(&mut body, x_ty, Expr::Construct(x_ty, comps))
        };
        basis.push(e_i);
    }
    for &e_i in &basis {
        args.push(pass(&mut g, &mut body, e_i, x_ty, d_params[extra_idx].by_ref));
        extra_idx += 1;
    }
    for &e_i in &basis {
        args.push(pass(&mut g, &mut body, e_i, x_ty, d_params[extra_idx].by_ref));
        extra_idx += 1;
    }
    args.push(pass(&mut g, &mut body, one, f32_ty, d_params[extra_idx].by_ref));
    let rt = m.functions[derived.index()]
        .ret
        .ok_or_else(|| Error::internal("no lipschitz return type"))?;
    let r = g.let_(&mut body, rt, Expr::Call(derived, args));
    let lip_field = 2 + 2 * n as u32;
    let lip = g.let_(&mut body, f32_ty, Expr::Extract(r, lip_field));
    let big = g.let_(&mut body, f32_ty, Expr::Const(Const::F32(BIG as f32)));
    let ok = g.let_(&mut body, m.types.bool(), Expr::Binary(BinOp::Le, lip, big));
    let out = g.let_(&mut body, f32_ty, Expr::Select { cond: ok, if_true: lip, if_false: big });
    body.push(Stmt::Return(Some(out)));
    g.body = body;
    Ok(m.add_function(g))
}

fn derive(
    m: &mut Module,
    cache: &mut DeriveCache,
    f: FuncId,
    mask: Vec<Bits>,
    n: usize,
    target: Target,
) -> R<FuncId> {
    let key = (f, mask.clone(), n as u8);
    if let Some(&id) = cache.lipschitz.get(&key) {
        return Ok(id);
    }
    if cache.lipschitz_open.contains(&key) {
        return Err(Error::not_derivable("a recursive function's Lipschitz bound"));
    }
    let orig = m.functions[f.index()].clone();
    if orig.body.is_empty() {
        return Err(Error::not_derivable("a function without a body"));
    }
    no_gpu_or_heap_work(&orig.body)?;
    let returns = returns_active(m, cache, f, mask.clone(), Mode::Interval) != 0;
    let (func, result, act) = one_exit(m, cache, f, orig, &mask);
    if act.active_loop_exit.is_some() {
        return Err(Error::not_derivable("a loop whose exit depends on the input"));
    }
    no_heap_store(&act)?;
    no_write_through_mut(&func, &act, &mask, "a Lipschitz bound")?;

    let bool_ty = m.types.bool();
    let f32_ty = m.types.f32();
    let mut nf = func.derived_signature();
    nf.name = format!("{}_lip{n}", func.name);

    let mut pparts = vec![Vec::new(); func.params.len()];
    for (i, p) in func.params.iter().enumerate() {
        if mask[i] == 0 {
            continue;
        }
        if p.ty == bool_ty {
            let mut add = |suf: &str| {
                pparts[i].push(nf.params.len() as u32);
                nf.params.push(Param {
                    name: format!("{}_{suf}", p.name),
                    ty: bool_ty,
                    by_ref: p.by_ref,
                    mutable: p.mutable,
                });
            };
            add("t");
            add("f");
            continue;
        }
        if !floaty(&m.types, p.ty) {
            return Err(Error::not_derivable("an active parameter that isn't made of floats"));
        }
        let mut add_p = |name: String, ty: TypeId| {
            pparts[i].push(nf.params.len() as u32);
            nf.params.push(Param { name, ty, by_ref: p.by_ref, mutable: p.mutable });
        };
        add_p(format!("{}_hi", p.name), p.ty);
        for k in 0..n {
            add_p(format!("{}_jl{k}", p.name), p.ty);
        }
        for k in 0..n {
            add_p(format!("{}_jh{k}", p.name), p.ty);
        }
        add_p(format!("{}_lip", p.name), f32_ty);
    }

    let mut lparts = vec![Vec::new(); func.locals.len()];
    for (l, decl) in func.locals.iter().enumerate() {
        if act.locals[l] == 0 {
            continue;
        }
        if decl.ty == bool_ty {
            lparts[l].push(nf.new_local(format!("{}_t", decl.name), bool_ty));
            lparts[l].push(nf.new_local(format!("{}_f", decl.name), bool_ty));
            continue;
        }
        if !floaty(&m.types, decl.ty) {
            return Err(Error::not_derivable("an active local that isn't made of floats"));
        }
        lparts[l].push(nf.new_local(format!("{}_hi", decl.name), decl.ty));
        for k in 0..n {
            lparts[l].push(nf.new_local(format!("{}_jl{k}", decl.name), decl.ty));
        }
        for k in 0..n {
            lparts[l].push(nf.new_local(format!("{}_jh{k}", decl.name), decl.ty));
        }
        lparts[l].push(nf.new_local(format!("{}_lip", decl.name), f32_ty));
    }

    if returns {
        let r = func.ret.ok_or_else(|| Error::internal("missing result type"))?;
        if !floaty(&m.types, r) || func.ret_ref {
            return Err(Error::not_derivable("an active result that isn't made of floats"));
        }
        let mut fields = vec![("lo".to_string(), r), ("hi".to_string(), r)];
        for k in 0..n {
            fields.push((format!("jl{k}"), r));
        }
        for k in 0..n {
            fields.push((format!("jh{k}"), r));
        }
        fields.push(("lip".to_string(), f32_ty));
        nf.ret_ref = false;
        nf.ret =
            Some(m.types.intern(TypeDef::Struct {
                name: format!("lip{n}<{}>", m.types.display(r)),
                fields,
            }));
    }

    cache.lipschitz_open.push(key.clone());
    let ptrs = Pointers::of(m, &func);
    let mut b = Lip {
        m,
        cache,
        f: &func,
        nf,
        act,
        n,
        target,
        bool_ty,
        f32_ty,
        pparts,
        lparts,
        vals: HashMap::new(),
        sure: HashMap::new(),
        meeting: HashSet::new(),
        returns,
        speculative: 0,
        ptrs,
    };
    let built = (|| -> R<Block> {
        let mut body = Vec::new();
        if let Some(r) = result.filter(|r| !b.lparts[r.index()].is_empty()) {
            let ty = b.f.locals[r.index()].ty;
            let e = b.empty(&mut body, ty);
            b.store_lval(&mut body, &Place::local(r), &e)?;
        }
        b.block_into(&func.body, &mut body)?;
        Ok(body)
    })();
    if let Some(i) = b.cache.lipschitz_open.iter().rposition(|k| *k == key) {
        b.cache.lipschitz_open.remove(i);
    }
    let mut nf = b.nf;
    nf.body = built?;
    let id = m.add_function(nf);
    cache.lipschitz.insert(key, id);
    Ok(id)
}

struct Lip<'a> {
    m: &'a mut Module,
    cache: &'a mut DeriveCache,
    f: &'a Function,
    nf: Function,
    act: Activity,
    n: usize,
    target: Target,
    bool_ty: TypeId,
    f32_ty: TypeId,
    pparts: Vec<Vec<u32>>,
    lparts: Vec<Vec<LocalId>>,
    vals: HashMap<ValueId, LVal>,
    sure: HashMap<ValueId, Sure>,
    /// The conditions made by `std::field::meets`: a branch on one is trusted to meet itself.
    meeting: HashSet<ValueId>,
    returns: bool,
    speculative: u32,
    ptrs: Pointers,
}

impl Lip<'_> {
    fn emit(&mut self, out: &mut Block, ty: TypeId, e: Expr) -> ValueId {
        self.nf.let_(out, ty, e)
    }

    fn ty(&self, v: ValueId) -> TypeId {
        self.nf.value_ty(v)
    }

    fn width(&self, ty: TypeId) -> Option<u8> {
        match self.m.types.get(ty) {
            &TypeDef::Vector(_, k) => Some(k),
            _ => None,
        }
    }

    fn gpu(&self) -> bool {
        self.target == Target::Gpu
    }

    fn tiny(&self) -> f64 {
        tiny(self.target)
    }

    fn bin(&mut self, out: &mut Block, op: BinOp, a: ValueId, b: ValueId, ty: TypeId) -> ValueId {
        let a = self.coerce(out, a, ty);
        let b = self.coerce(out, b, ty);
        self.emit(out, ty, Expr::Binary(op, a, b))
    }

    fn builtin(&mut self, out: &mut Block, b: Builtin, args: Vec<ValueId>, ty: TypeId) -> ValueId {
        self.emit(out, ty, Expr::Builtin(b, args))
    }

    fn coerce(&mut self, out: &mut Block, v: ValueId, ty: TypeId) -> ValueId {
        if self.ty(v) == ty {
            return v;
        }
        if let TypeDef::Vector(_, k) = *self.m.types.get(ty) {
            self.emit(out, ty, Expr::Splat(v, k))
        } else {
            v
        }
    }

    fn k(&mut self, out: &mut Block, ty: TypeId, x: f64) -> ValueId {
        match self.m.types.get(ty).clone() {
            TypeDef::Struct { fields, .. } => {
                let parts = fields.iter().map(|(_, ft)| self.k(out, *ft, x)).collect();
                self.emit(out, ty, Expr::Construct(ty, parts))
            }
            TypeDef::Array(et, n) => {
                let elem = self.k(out, et, x);
                self.emit(out, ty, Expr::Construct(ty, vec![elem; n as usize]))
            }
            _ => {
                let s = self.emit(out, self.f32_ty, Expr::Const(Const::F32(x as f32)));
                self.coerce(out, s, ty)
            }
        }
    }

    fn kf(&mut self, out: &mut Block, x: f64) -> ValueId {
        self.emit(out, self.f32_ty, Expr::Const(Const::F32(x as f32)))
    }

    fn bconst(&mut self, out: &mut Block, b: bool) -> ValueId {
        self.emit(out, self.bool_ty, Expr::Const(Const::Bool(b)))
    }

    fn zero(&mut self, out: &mut Block, ty: TypeId) -> ValueId {
        self.emit(out, ty, Expr::Zero(ty))
    }

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
        if self.ty(cond) == self.bool_ty {
            return self.emit(out, ty, Expr::Select { cond, if_true: t, if_false: f });
        }
        let k = self.width(ty).expect("vector select");
        let elem = match self.m.types.get(ty) {
            &TypeDef::Vector(s, _) => self.m.types.intern(TypeDef::Scalar(s)),
            _ => unreachable!(),
        };
        let mut parts = Vec::with_capacity(k as usize);
        for j in 0..k as u32 {
            let cj = self.emit(out, self.bool_ty, Expr::Extract(cond, j));
            let tj = self.emit(out, elem, Expr::Extract(t, j));
            let fj = self.emit(out, elem, Expr::Extract(f, j));
            parts.push(self.emit(out, elem, Expr::Select { cond: cj, if_true: tj, if_false: fj }));
        }
        self.emit(out, ty, Expr::Construct(ty, parts))
    }

    /// Clamps a non-negative `f32` bound upward and turns NaN/Inf into `BIG`.
    fn cap_lip(&mut self, out: &mut Block, x: ValueId) -> ValueId {
        let f = self.f32_ty;
        let big = self.kf(out, BIG);
        let ok = self.cmp(out, BinOp::Le, x, big);
        let x = self.sel(out, ok, x, big);
        let z = self.kf(out, 0.0);
        self.builtin(out, Builtin::Max, vec![x, z], f)
    }

    /// Pads a positive `f32` upper bound upward by `ulps * EPS` + `tiny`, clamped at `BIG`.
    fn pad_up(&mut self, out: &mut Block, x: ValueId, ulps: f64) -> ValueId {
        let f = self.f32_ty;
        let u = if self.gpu() { 2.0 * ulps } else { ulps };
        let s = self.kf(out, 1.0 + u * EPS);
        let t = self.kf(out, self.tiny());
        let y = self.bin(out, BinOp::Mul, x, s, f);
        let y = self.bin(out, BinOp::Add, y, t, f);
        self.cap_lip(out, y)
    }

    /// Pads an interval `[lo, hi]` outward by `ulps * EPS` of magnitude + `tiny`, clamped to `±BIG`.
    fn pad_iv(
        &mut self,
        out: &mut Block,
        lo: ValueId,
        hi: ValueId,
        ty: TypeId,
        ulps: f64,
    ) -> (ValueId, ValueId) {
        let al = self.builtin(out, Builtin::Abs, vec![lo], ty);
        let ah = self.builtin(out, Builtin::Abs, vec![hi], ty);
        let m = self.builtin(out, Builtin::Max, vec![al, ah], ty);
        let u = if self.gpu() { 2.0 * ulps } else { ulps };
        let k = self.k(out, ty, u * EPS);
        let t = self.k(out, ty, self.tiny());
        let p = self.bin(out, BinOp::Mul, m, k, ty);
        let p = self.bin(out, BinOp::Add, p, t, ty);
        let lo = self.bin(out, BinOp::Sub, lo, p, ty);
        let hi = self.bin(out, BinOp::Add, hi, p, ty);
        self.clamp_iv(out, lo, hi, ty)
    }

    fn clamp_iv(
        &mut self,
        out: &mut Block,
        lo: ValueId,
        hi: ValueId,
        ty: TypeId,
    ) -> (ValueId, ValueId) {
        let big = self.k(out, ty, BIG);
        let nbig = self.k(out, ty, -BIG);
        let lo_ok = self.cmp(out, BinOp::Ge, lo, nbig);
        let lo_ok2 = self.cmp(out, BinOp::Le, lo, big);
        let bt = self.bool_of(ty);
        let lo_fin = self.emit(out, bt, Expr::Binary(BinOp::And, lo_ok, lo_ok2));
        let lo = self.sel(out, lo_fin, lo, nbig);
        let hi_ok = self.cmp(out, BinOp::Ge, hi, nbig);
        let hi_ok2 = self.cmp(out, BinOp::Le, hi, big);
        let hi_fin = self.emit(out, bt, Expr::Binary(BinOp::And, hi_ok, hi_ok2));
        let hi = self.sel(out, hi_fin, hi, big);
        (lo, hi)
    }

    fn iv_add(
        &mut self,
        out: &mut Block,
        a: (ValueId, ValueId),
        b: (ValueId, ValueId),
        ty: TypeId,
    ) -> (ValueId, ValueId) {
        let lo = self.bin(out, BinOp::Add, a.0, b.0, ty);
        let hi = self.bin(out, BinOp::Add, a.1, b.1, ty);
        self.pad_iv(out, lo, hi, ty, 2.0)
    }

    fn iv_sub(
        &mut self,
        out: &mut Block,
        a: (ValueId, ValueId),
        b: (ValueId, ValueId),
        ty: TypeId,
    ) -> (ValueId, ValueId) {
        let lo = self.bin(out, BinOp::Sub, a.0, b.1, ty);
        let hi = self.bin(out, BinOp::Sub, a.1, b.0, ty);
        self.pad_iv(out, lo, hi, ty, 2.0)
    }

    fn iv_neg(&mut self, out: &mut Block, a: (ValueId, ValueId), ty: TypeId) -> (ValueId, ValueId) {
        let lo = self.emit(out, ty, Expr::Unary(UnOp::Neg, a.1));
        let hi = self.emit(out, ty, Expr::Unary(UnOp::Neg, a.0));
        (lo, hi)
    }

    fn iv_hull(
        &mut self,
        out: &mut Block,
        a: (ValueId, ValueId),
        b: (ValueId, ValueId),
        ty: TypeId,
    ) -> (ValueId, ValueId) {
        let lo = self.builtin(out, Builtin::Min, vec![a.0, b.0], ty);
        let hi = self.builtin(out, Builtin::Max, vec![a.1, b.1], ty);
        (lo, hi)
    }

    fn iv_mul(
        &mut self,
        out: &mut Block,
        a: (ValueId, ValueId),
        b: (ValueId, ValueId),
        ty: TypeId,
    ) -> (ValueId, ValueId) {
        let cap = self.k(out, ty, CAP);
        let ncap = self.k(out, ty, -CAP);
        let bt = self.bool_of(ty);
        let ok_iv = |s: &mut Self, out: &mut Block, (l, h): (ValueId, ValueId)| -> ValueId {
            let a1 = s.cmp(out, BinOp::Ge, l, ncap);
            let a2 = s.cmp(out, BinOp::Le, h, cap);
            s.emit(out, bt, Expr::Binary(BinOp::And, a1, a2))
        };
        let oka = ok_iv(self, out, a);
        let okb = ok_iv(self, out, b);
        let ok = self.emit(out, bt, Expr::Binary(BinOp::And, oka, okb));
        let z = self.zero(out, ty);
        let a0 = self.sel(out, ok, a.0, z);
        let a1 = self.sel(out, ok, a.1, z);
        let b0 = self.sel(out, ok, b.0, z);
        let b1 = self.sel(out, ok, b.1, z);
        let p00 = self.bin(out, BinOp::Mul, a0, b0, ty);
        let p01 = self.bin(out, BinOp::Mul, a0, b1, ty);
        let p10 = self.bin(out, BinOp::Mul, a1, b0, ty);
        let p11 = self.bin(out, BinOp::Mul, a1, b1, ty);
        let lo0 = self.builtin(out, Builtin::Min, vec![p00, p01], ty);
        let lo1 = self.builtin(out, Builtin::Min, vec![p10, p11], ty);
        let lo = self.builtin(out, Builtin::Min, vec![lo0, lo1], ty);
        let hi0 = self.builtin(out, Builtin::Max, vec![p00, p01], ty);
        let hi1 = self.builtin(out, Builtin::Max, vec![p10, p11], ty);
        let hi = self.builtin(out, Builtin::Max, vec![hi0, hi1], ty);
        let (lo, hi) = self.pad_iv(out, lo, hi, ty, 3.0);
        let big = self.k(out, ty, BIG);
        let nbig = self.k(out, ty, -BIG);
        (self.sel(out, ok, lo, nbig), self.sel(out, ok, hi, big))
    }

    fn iv_abs(&mut self, out: &mut Block, a: (ValueId, ValueId), ty: TypeId) -> (ValueId, ValueId) {
        let z = self.zero(out, ty);
        let al = self.builtin(out, Builtin::Abs, vec![a.0], ty);
        let ah = self.builtin(out, Builtin::Abs, vec![a.1], ty);
        let hi = self.builtin(out, Builtin::Max, vec![al, ah], ty);
        let pos = self.cmp(out, BinOp::Ge, a.0, z);
        let neg = self.cmp(out, BinOp::Le, a.1, z);
        let lo_nz = self.sel(out, pos, a.0, ah);
        let bt = self.bool_of(ty);
        let any_nz = self.emit(out, bt, Expr::Binary(BinOp::Or, pos, neg));
        let lo = self.sel(out, any_nz, lo_nz, z);
        (lo, hi)
    }

    /// Maximum absolute value across components of `(lo, hi)` (returns `f32`).
    fn max_abs_scalar(&mut self, out: &mut Block, a: (ValueId, ValueId), ty: TypeId) -> ValueId {
        let (_, hi) = self.iv_abs(out, a, ty);
        match self.width(ty) {
            None => hi,
            Some(k) => {
                let f = self.f32_ty;
                let mut m = self.emit(out, f, Expr::Extract(hi, 0));
                for j in 1..k as u32 {
                    let c = self.emit(out, f, Expr::Extract(hi, j));
                    m = self.builtin(out, Builtin::Max, vec![m, c], f);
                }
                m
            }
        }
    }

    /// Euclidean upper bound on `||v||₂` for an interval `v = (lo, hi)`.
    fn vec_len_hi(&mut self, out: &mut Block, a: (ValueId, ValueId), ty: TypeId) -> ValueId {
        let f = self.f32_ty;
        let (_, hi) = self.iv_abs(out, a, ty);
        let cap = self.k(out, ty, CAP);
        let capped = self.builtin(out, Builtin::Min, vec![hi, cap], ty);
        let sq = self.bin(out, BinOp::Mul, capped, capped, ty);
        let sum = match self.width(ty) {
            None => sq,
            Some(k) => {
                let mut s = self.emit(out, f, Expr::Extract(sq, 0));
                for j in 1..k as u32 {
                    let c = self.emit(out, f, Expr::Extract(sq, j));
                    s = self.bin(out, BinOp::Add, s, c, f);
                }
                s
            }
        };
        let len = self.builtin(out, Builtin::Sqrt, vec![sum], f);
        let len = self.pad_up(out, len, 8.0);
        let m = self.max_abs_scalar(out, a, ty);
        let cap_f = self.kf(out, CAP);
        let ok = self.cmp(out, BinOp::Le, m, cap_f);
        let big = self.kf(out, BIG);
        self.sel(out, ok, len, big)
    }

    /// Euclidean lower bound on `||v||₂` for an interval `v = (lo, hi)`.
    fn vec_len_lo(&mut self, out: &mut Block, a: (ValueId, ValueId), ty: TypeId) -> ValueId {
        let f = self.f32_ty;
        let (lo, _) = self.iv_abs(out, a, ty);
        let cap = self.k(out, ty, CAP);
        let capped = self.builtin(out, Builtin::Min, vec![lo, cap], ty);
        let sq = self.bin(out, BinOp::Mul, capped, capped, ty);
        let sum = match self.width(ty) {
            None => sq,
            Some(k) => {
                let mut s = self.emit(out, f, Expr::Extract(sq, 0));
                for j in 1..k as u32 {
                    let c = self.emit(out, f, Expr::Extract(sq, j));
                    s = self.bin(out, BinOp::Add, s, c, f);
                }
                s
            }
        };
        let len = self.builtin(out, Builtin::Sqrt, vec![sum], f);
        let dn = self.kf(out, 1.0 - 8.0 * EPS);
        let z = self.kf(out, 0.0);
        let len = self.bin(out, BinOp::Mul, len, dn, f);
        self.builtin(out, Builtin::Max, vec![len, z], f)
    }

    /// Upper bound on `||J||₂` from interval Jacobian columns `(j_lo, j_hi)`:
    /// - scalar (`f32`): exact `√(Σ max(|g_lo_i|, |g_hi_i|)²)`
    /// - vector (`vecK`): Gershgorin bound on `G = J Jᵀ` (exact `1.0` for any orthogonal rotation
    ///   matrix `R` or diagonal `±1` matrix `diag(±1, ...)`!).
    fn op_norm_hi(
        &mut self,
        out: &mut Block,
        j_lo: &[ValueId],
        j_hi: &[ValueId],
        ty: TypeId,
    ) -> ValueId {
        self.op_norm_hi_rows(out, j_lo, j_hi, ty, None)
    }

    fn op_norm_hi_rows(
        &mut self,
        out: &mut Block,
        j_lo: &[ValueId],
        j_hi: &[ValueId],
        ty: TypeId,
        row_lips: Option<&[ValueId]>,
    ) -> ValueId {
        let f = self.f32_ty;
        let Some(k) = self.width(ty) else {
            let mut sum = self.kf(out, 0.0);
            let mut ok = self.bconst(out, true);
            let cap = self.kf(out, CAP);
            for i in 0..self.n {
                let (_, hi) = self.iv_abs(out, (j_lo[i], j_hi[i]), f);
                let ok_i = self.cmp(out, BinOp::Le, hi, cap);
                ok = self.emit(out, self.bool_ty, Expr::Binary(BinOp::And, ok, ok_i));
                let hc = self.builtin(out, Builtin::Min, vec![hi, cap], f);
                let sq = self.bin(out, BinOp::Mul, hc, hc, f);
                sum = self.bin(out, BinOp::Add, sum, sq, f);
            }
            let nrm = self.builtin(out, Builtin::Sqrt, vec![sum], f);
            let nrm = self.pad_up(out, nrm, 8.0);
            let big = self.kf(out, BIG);
            return self.sel(out, ok, nrm, big);
        };
        let k = k as usize;
        // Extract rows `r in 0..k`, each having `n` scalar intervals `(lo, hi)`.
        let mut rows: Vec<Vec<(ValueId, ValueId)>> = Vec::with_capacity(k);
        for r in 0..k {
            let mut row = Vec::with_capacity(self.n);
            for i in 0..self.n {
                let rl = self.emit(out, f, Expr::Extract(j_lo[i], r as u32));
                let rh = self.emit(out, f, Expr::Extract(j_hi[i], r as u32));
                row.push((rl, rh));
            }
            rows.push(row);
        }
        // Symmetric Gram upper bounds `G_{r,s} >= |dot(row_r, row_s)|`.
        // On the diagonal, `G_{r,r} = ||row_r||₂² <= row_lips[r]²` when per-component Lipschitz
        // bounds are known (keeps `vec2(length(p.xz), p.y)` exact `1.0` even on the axis!).
        let cap = self.kf(out, CAP);
        let mut g = vec![vec![self.kf(out, 0.0); k]; k];
        for r in 0..k {
            for s in r..k {
                let mut acc = (self.kf(out, 0.0), self.kf(out, 0.0));
                for i in 0..self.n {
                    let p = self.iv_mul(out, rows[r][i], rows[s][i], f);
                    acc = self.iv_add(out, acc, p, f);
                }
                let (_, mut abs_hi) = self.iv_abs(out, acc, f);
                if r == s
                    && let Some(rl) = row_lips
                {
                    let c = self.builtin(out, Builtin::Min, vec![rl[r], cap], f);
                    let c2 = self.bin(out, BinOp::Mul, c, c, f);
                    let c2 = self.pad_up(out, c2, 4.0);
                    abs_hi = self.builtin(out, Builtin::Min, vec![abs_hi, c2], f);
                }
                g[r][s] = abs_hi;
                g[s][r] = abs_hi;
            }
        }
        // Gershgorin bound on largest eigenvalue of `J Jᵀ`: `max_r Σ_s G_{r,s}`.
        let mut max_row = self.kf(out, 0.0);
        for r in 0..k {
            let mut row_sum = g[r][0];
            for s in 1..k {
                row_sum = self.bin(out, BinOp::Add, row_sum, g[r][s], f);
            }
            max_row = self.builtin(out, Builtin::Max, vec![max_row, row_sum], f);
        }
        let gersh = self.builtin(out, Builtin::Sqrt, vec![max_row], f);
        self.pad_up(out, gersh, 16.0)
    }

    /// Tightens scalar `lip` against `op_norm_hi(j_lo, j_hi)`.
    fn tighten(
        &mut self,
        out: &mut Block,
        lip: ValueId,
        j_lo: &[ValueId],
        j_hi: &[ValueId],
        ty: TypeId,
    ) -> ValueId {
        let j_nrm = self.op_norm_hi(out, j_lo, j_hi, ty);
        let lip = self.cap_lip(out, lip);
        self.builtin(out, Builtin::Min, vec![lip, j_nrm], self.f32_ty)
    }

    /// Whether every component of `c`, a comparison of two values of type `ty`, holds.
    fn all_of(&mut self, out: &mut Block, c: ValueId, ty: TypeId) -> ValueId {
        let Some(k) = self.width(ty) else {
            return c;
        };
        let bt = self.bool_ty;
        let mut acc = self.emit(out, bt, Expr::Extract(c, 0));
        for j in 1..k as u32 {
            let cj = self.emit(out, bt, Expr::Extract(c, j));
            acc = self.emit(out, bt, Expr::Binary(BinOp::And, acc, cj));
        }
        acc
    }

    /// A step function that never falls (`floor`, `ceil`, `round`, `trunc`, `sign`, `step` past
    /// an edge the input doesn't move), `fl` and `fh` at the box's two ends: constant over the
    /// box where they agree (no slope), and where they don't, a step lies inside it, where no
    /// Lipschitz bound holds (`BIG`, its Jacobian unbounded too).
    fn stepwise(&mut self, out: &mut Block, fl: ValueId, fh: ValueId, ty: TypeId) -> LVal {
        let same = self.cmp(out, BinOp::Eq, fl, fh);
        let flat = self.all_of(out, same, ty);
        let z = self.zero(out, ty);
        let big = self.k(out, ty, BIG);
        let nbig = self.k(out, ty, -BIG);
        let jl = self.sel(out, same, z, nbig);
        let jh = self.sel(out, same, z, big);
        let zf = self.kf(out, 0.0);
        let bigf = self.kf(out, BIG);
        let lip = self.sel(out, flat, zf, bigf);
        LVal { lo: fl, hi: fh, j_lo: vec![jl; self.n], j_hi: vec![jh; self.n], lip }
    }

    fn exact(&mut self, out: &mut Block, v: ValueId) -> LVal {
        let ty = self.ty(v);
        let z = self.zero(out, ty);
        let zf = self.kf(out, 0.0);
        LVal { lo: v, hi: v, j_lo: vec![z; self.n], j_hi: vec![z; self.n], lip: zf }
    }

    fn empty(&mut self, out: &mut Block, ty: TypeId) -> LVal {
        let big = self.k(out, ty, BIG);
        let nbig = self.k(out, ty, -BIG);
        let zf = self.kf(out, 0.0);
        LVal { lo: big, hi: nbig, j_lo: vec![big; self.n], j_hi: vec![nbig; self.n], lip: zf }
    }

    fn every(&mut self, out: &mut Block, ty: TypeId) -> LVal {
        let big = self.k(out, ty, BIG);
        let nbig = self.k(out, ty, -BIG);
        let bigf = self.kf(out, BIG);
        LVal { lo: nbig, hi: big, j_lo: vec![nbig; self.n], j_hi: vec![big; self.n], lip: bigf }
    }

    fn of(&mut self, out: &mut Block, v: ValueId) -> LVal {
        match self.vals.get(&v) {
            Some(a) => a.clone(),
            None => self.exact(out, v),
        }
    }

    fn of_as(&mut self, out: &mut Block, v: ValueId, ty: TypeId) -> LVal {
        let a = self.of(out, v);
        if self.ty(a.lo) == ty {
            return a;
        }
        let Some(k) = self.width(ty) else {
            return a;
        };
        let lo = self.emit(out, ty, Expr::Splat(a.lo, k));
        let hi = self.emit(out, ty, Expr::Splat(a.hi, k));
        let j_lo = a.j_lo.iter().map(|&x| self.emit(out, ty, Expr::Splat(x, k))).collect();
        let j_hi = a.j_hi.iter().map(|&x| self.emit(out, ty, Expr::Splat(x, k))).collect();
        let sk = self.kf(out, (k as f64).sqrt());
        let lip = self.bin(out, BinOp::Mul, a.lip, sk, self.f32_ty);
        let lip = self.pad_up(out, lip, 4.0);
        LVal { lo, hi, j_lo, j_hi, lip }
    }

    fn pick(&mut self, out: &mut Block, cond: ValueId, t: &LVal, f: &LVal) -> LVal {
        let lo = self.sel(out, cond, t.lo, f.lo);
        let hi = self.sel(out, cond, t.hi, f.hi);
        let j_lo = (0..self.n).map(|i| self.sel(out, cond, t.j_lo[i], f.j_lo[i])).collect();
        let j_hi = (0..self.n).map(|i| self.sel(out, cond, t.j_hi[i], f.j_hi[i])).collect();
        let lip = self.sel(out, cond, t.lip, f.lip);
        LVal { lo, hi, j_lo, j_hi, lip }
    }

    /// `t` where the box surely has the condition, `f` where it surely hasn't, and across the
    /// switch their values' hull: unbounded (the result may jump there: the module's doc) but
    /// for a condition `std::field::meets` made (`meets`), whose two sides the author states
    /// agree at the switch, where the bound is the steeper side's and the Jacobian their hull.
    fn join(
        &mut self,
        out: &mut Block,
        (st, sf): Sure,
        meets: bool,
        t: &LVal,
        f: &LVal,
        ty: TypeId,
    ) -> LVal {
        let (lo, hi) = self.iv_hull(out, (t.lo, t.hi), (f.lo, f.hi), ty);
        let h = if meets {
            let mut j_lo = Vec::with_capacity(self.n);
            let mut j_hi = Vec::with_capacity(self.n);
            for i in 0..self.n {
                let (jl, jh) =
                    self.iv_hull(out, (t.j_lo[i], t.j_hi[i]), (f.j_lo[i], f.j_hi[i]), ty);
                j_lo.push(jl);
                j_hi.push(jh);
            }
            let lip = self.builtin(out, Builtin::Max, vec![t.lip, f.lip], self.f32_ty);
            LVal { lo, hi, j_lo, j_hi, lip }
        } else {
            let big = self.k(out, ty, BIG);
            let nbig = self.k(out, ty, -BIG);
            let lip = self.kf(out, BIG);
            LVal { lo, hi, j_lo: vec![nbig; self.n], j_hi: vec![big; self.n], lip }
        };
        let r = self.pick(out, sf, f, &h);
        self.pick(out, st, t, &r)
    }

    fn join_bool(&mut self, out: &mut Block, (st, sf): Sure, t: Sure, f: Sure) -> Sure {
        let bt = self.bool_ty;
        let ht = self.emit(out, bt, Expr::Binary(BinOp::And, t.0, f.0));
        let hf = self.emit(out, bt, Expr::Binary(BinOp::And, t.1, f.1));
        let r0 = self.sel(out, sf, f.0, ht);
        let r1 = self.sel(out, sf, f.1, hf);
        (self.sel(out, st, t.0, r0), self.sel(out, st, t.1, r1))
    }

    fn sure_of(&mut self, out: &mut Block, v: ValueId) -> Sure {
        if let Some(&s) = self.sure.get(&v) {
            return s;
        }
        let not = self.emit(out, self.bool_ty, Expr::Unary(UnOp::Not, v));
        (v, not)
    }

    // ---- Places & control flow -------------------------------------------------------------

    fn active_root(&self, r: &PlaceRoot) -> bool {
        match r {
            PlaceRoot::Local(l) => !self.lparts[l.index()].is_empty(),
            PlaceRoot::Param(i) => !self.pparts[*i as usize].is_empty(),
            _ => false,
        }
    }

    fn is_bool_root(&self, r: &PlaceRoot) -> bool {
        match r {
            PlaceRoot::Local(l) => self.f.locals[l.index()].ty == self.bool_ty,
            PlaceRoot::Param(i) => self.f.params[*i as usize].ty == self.bool_ty,
            _ => false,
        }
    }

    fn root_ty(&self, r: &PlaceRoot) -> R<TypeId> {
        match r {
            PlaceRoot::Local(l) => Ok(self.f.locals[l.index()].ty),
            PlaceRoot::Param(i) => Ok(self.f.params[*i as usize].ty),
            _ => Err(Error::internal("joined place that isn't a local or parameter")),
        }
    }

    fn place_parts(&self, p: &Place) -> R<(Vec<Place>, Place)> {
        let roots: Vec<PlaceRoot> = match p.root {
            PlaceRoot::Local(l) => {
                self.lparts[l.index()].iter().map(|&k| PlaceRoot::Local(k)).collect()
            }
            PlaceRoot::Param(i) => {
                self.pparts[i as usize].iter().map(|&k| PlaceRoot::Param(k)).collect()
            }
            _ => return Err(Error::internal("inactive place has no parts")),
        };
        let mut all: Vec<Place> = roots
            .iter()
            .take(roots.len().saturating_sub(1))
            .map(|r| Place { root: r.clone(), path: p.path.clone() })
            .collect();
        let last = roots.last().ok_or_else(|| Error::internal("missing lip part"))?.clone();
        // `lip` is always a scalar `f32` at root (no path projection).
        let lip_place = Place::root(last);
        if self.is_bool_root(&p.root) {
            all.push(Place { root: lip_place.root.clone(), path: p.path.clone() });
        }
        Ok((all, lip_place))
    }

    fn load_lval(&mut self, out: &mut Block, p: &Place, ty: TypeId) -> R<LVal> {
        let (parts, lip_p) = self.place_parts(p)?;
        let lo = self.emit(out, ty, Expr::Load(p.clone()));
        let hi = self.emit(out, ty, Expr::Load(parts[0].clone()));
        let j_lo =
            (0..self.n).map(|i| self.emit(out, ty, Expr::Load(parts[1 + i].clone()))).collect();
        let j_hi = (0..self.n)
            .map(|i| self.emit(out, ty, Expr::Load(parts[1 + self.n + i].clone())))
            .collect();
        let lip = self.emit(out, self.f32_ty, Expr::Load(lip_p));
        Ok(LVal { lo, hi, j_lo, j_hi, lip })
    }

    fn store_lval(&mut self, out: &mut Block, p: &Place, a: &LVal) -> R<()> {
        let (parts, lip_p) = self.place_parts(p)?;
        out.push(Stmt::Store(p.clone(), a.lo));
        out.push(Stmt::Store(parts[0].clone(), a.hi));
        for i in 0..self.n {
            out.push(Stmt::Store(parts[1 + i].clone(), a.j_lo[i]));
            out.push(Stmt::Store(parts[1 + self.n + i].clone(), a.j_hi[i]));
        }
        if p.path.is_empty() {
            out.push(Stmt::Store(lip_p, a.lip));
        } else {
            let prev = self.emit(out, self.f32_ty, Expr::Load(lip_p.clone()));
            let next = self.builtin(out, Builtin::Max, vec![prev, a.lip], self.f32_ty);
            out.push(Stmt::Store(lip_p, next));
        }
        Ok(())
    }

    fn load_sure(&mut self, out: &mut Block, p: &Place) -> R<Sure> {
        let (parts, _) = self.place_parts(p)?;
        let t = self.emit(out, self.bool_ty, Expr::Load(parts[0].clone()));
        let f = self.emit(out, self.bool_ty, Expr::Load(parts[1].clone()));
        Ok((t, f))
    }

    fn store_sure(&mut self, out: &mut Block, p: &Place, (t, f): Sure) -> R<()> {
        let (parts, _) = self.place_parts(p)?;
        out.push(Stmt::Store(parts[0].clone(), t));
        out.push(Stmt::Store(parts[1].clone(), f));
        Ok(())
    }

    fn load_state(&mut self, out: &mut Block, r: &PlaceRoot) -> R<LState> {
        let p = Place::root(r.clone());
        if self.is_bool_root(r) {
            return Ok(LState::Bool(self.load_sure(out, &p)?));
        }
        let ty = self.root_ty(r)?;
        Ok(LState::Float(self.load_lval(out, &p, ty)?))
    }

    fn store_state(&mut self, out: &mut Block, r: &PlaceRoot, s: &LState) -> R<()> {
        let p = Place::root(r.clone());
        match s {
            LState::Bool(b) => self.store_sure(out, &p, *b),
            LState::Float(a) => self.store_lval(out, &p, a),
        }
    }

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

    fn reaches_active(&self, r: &PlaceRoot) -> bool {
        let PlaceRoot::Ptr(v) = r else { return false };
        let mut roots = Vec::new();
        if self.ptrs.roots(*v, &mut roots).is_none() {
            return self.lparts.iter().any(|p| !p.is_empty())
                || self.pparts.iter().any(|p| !p.is_empty());
        }
        roots.iter().any(|r| self.active_root(r))
    }

    fn lends_through_pointer(&self, g: FuncId, args: &[Arg]) -> bool {
        args.iter()
            .zip(&self.m.functions[g.index()].params)
            .any(|(a, p)| matches!(a, Arg::Place(pl) if p.mutable && self.reaches_active(&pl.root)))
    }

    fn any_active(&self, args: &[Arg]) -> bool {
        args.iter().any(|a| arg_bits(self.m, self.f, &self.act, a) != 0)
    }

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
            Stmt::Let(v, Expr::Extract(x, k)) if self.vals.contains_key(x) => {
                let lo = self.vals[x].lo;
                out.push(Stmt::Let(*v, Expr::Extract(lo, *k)));
                Ok(())
            }
            Stmt::Let(v, Expr::Swizzle(x, sw)) if self.vals.contains_key(x) => {
                let lo = self.vals[x].lo;
                out.push(Stmt::Let(*v, Expr::Swizzle(lo, sw.clone())));
                Ok(())
            }
            Stmt::Store(p, v) if self.active_root(&p.root) => {
                if self.is_bool_root(&p.root) {
                    let s = self.sure_of(out, *v);
                    return self.store_sure(out, p, s);
                }
                let ty = place_type(self.m, self.f, p)?;
                let a = self.of_as(out, *v, ty);
                self.store_lval(out, p, &a)
            }
            Stmt::Store(_, v) if self.act.value(*v) => {
                Err(Error::not_derivable("a store of what depends on the input into another place"))
            }
            Stmt::Store(p, _) if self.reaches_active(&p.root) => Err(Error::not_derivable(
                "a store through a pointer that may reach what depends on the input",
            )),
            Stmt::If { cond, then, else_ } if self.act.value(*cond) => {
                let sure = self.sure_of(out, *cond);
                let meets = self.meeting.contains(cond);
                let mut roots = Vec::new();
                self.written(then, &mut roots);
                self.written(else_, &mut roots);
                for r in &roots {
                    let t = self.root_ty(r)?;
                    if !self.active_root(r) || (t != self.bool_ty && !is_float(&self.m.types, t)) {
                        return Err(Error::not_derivable(
                            "a branch on the input that writes a non-float place",
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
                        (LState::Float(t), LState::Float(e)) => {
                            let ty = self.root_ty(r)?;
                            LState::Float(self.join(out, sure, meets, &t, &e, ty))
                        }
                        (LState::Bool(t), LState::Bool(e)) => {
                            LState::Bool(self.join_bool(out, sure, t, e))
                        }
                        _ => return Err(Error::internal("local state changed kind")),
                    };
                    self.store_state(out, r, &j)?;
                }
                Ok(())
            }
            Stmt::Return(Some(v)) if self.returns => {
                let ty = self.f.ret.ok_or_else(|| Error::internal("no result type"))?;
                let a = self.of_as(out, *v, ty);
                let rt = self.nf.ret.ok_or_else(|| Error::internal("no lipschitz return type"))?;
                let mut fields = vec![a.lo, a.hi];
                fields.extend(a.j_lo.iter().copied());
                fields.extend(a.j_hi.iter().copied());
                fields.push(a.lip);
                let r = self.emit(out, rt, Expr::Construct(rt, fields));
                out.push(Stmt::Return(Some(r)));
                Ok(())
            }
            Stmt::Trap if self.speculative > 0 => Ok(()),
            other => pass_through(other, out, &mut |b| self.block(b)),
        }
    }

    // ---- Expressions -----------------------------------------------------------------------

    fn let_(&mut self, v: ValueId, e: &Expr, out: &mut Block) -> R<()> {
        let ty = self.f.value_ty(v);
        if ty == self.bool_ty {
            let s = self.let_bool(e, out)?;
            self.sure.insert(v, s);
            return Ok(());
        }
        if !floaty(&self.m.types, ty) {
            return Err(Error::not_derivable("active value isn't made of floats or bool"));
        }
        let a = match e {
            Expr::Param(i) if !self.pparts[*i as usize].is_empty() => {
                let lo = self.emit(out, ty, Expr::Param(*i));
                let ps = self.pparts[*i as usize].clone();
                let hi = self.emit(out, ty, Expr::Param(ps[0]));
                let j_lo =
                    (0..self.n).map(|k| self.emit(out, ty, Expr::Param(ps[1 + k]))).collect();
                let j_hi = (0..self.n)
                    .map(|k| self.emit(out, ty, Expr::Param(ps[1 + self.n + k])))
                    .collect();
                let lip = self.emit(out, self.f32_ty, Expr::Param(*ps.last().unwrap()));
                LVal { lo, hi, j_lo, j_hi, lip }
            }
            Expr::Load(p) if self.active_root(&p.root) => self.load_lval(out, p, ty)?,
            Expr::Unary(UnOp::Neg, x) => {
                let a = self.of_as(out, *x, ty);
                let (lo, hi) = self.iv_neg(out, (a.lo, a.hi), ty);
                let mut j_lo = Vec::with_capacity(self.n);
                let mut j_hi = Vec::with_capacity(self.n);
                for i in 0..self.n {
                    let (jl, jh) = self.iv_neg(out, (a.j_lo[i], a.j_hi[i]), ty);
                    j_lo.push(jl);
                    j_hi.push(jh);
                }
                LVal { lo, hi, j_lo, j_hi, lip: a.lip }
            }
            Expr::Binary(op @ (BinOp::Add | BinOp::Sub), a, b) => {
                let (aa, bb) = (self.of_as(out, *a, ty), self.of_as(out, *b, ty));
                let sub = *op == BinOp::Sub;
                let (lo, hi) = if sub {
                    self.iv_sub(out, (aa.lo, aa.hi), (bb.lo, bb.hi), ty)
                } else {
                    self.iv_add(out, (aa.lo, aa.hi), (bb.lo, bb.hi), ty)
                };
                let mut j_lo = Vec::with_capacity(self.n);
                let mut j_hi = Vec::with_capacity(self.n);
                for i in 0..self.n {
                    let (jl, jh) = if sub {
                        self.iv_sub(out, (aa.j_lo[i], aa.j_hi[i]), (bb.j_lo[i], bb.j_hi[i]), ty)
                    } else {
                        self.iv_add(out, (aa.j_lo[i], aa.j_hi[i]), (bb.j_lo[i], bb.j_hi[i]), ty)
                    };
                    j_lo.push(jl);
                    j_hi.push(jh);
                }
                let sum_lip = self.bin(out, BinOp::Add, aa.lip, bb.lip, self.f32_ty);
                let sum_lip = self.pad_up(out, sum_lip, 2.0);
                let lip = self.tighten(out, sum_lip, &j_lo, &j_hi, ty);
                LVal { lo, hi, j_lo, j_hi, lip }
            }
            Expr::Binary(BinOp::Mul, a, b) => self.mul_expr(out, *a, *b, ty)?,
            Expr::Binary(BinOp::Div, a, b) => self.div_expr(out, *a, *b, ty),
            Expr::Select { cond, if_true, if_false } => {
                let (t, f) = (self.of_as(out, *if_true, ty), self.of_as(out, *if_false, ty));
                if self.act.value(*cond) {
                    let sure = self.sure_of(out, *cond);
                    let meets = self.meeting.contains(cond);
                    self.join(out, sure, meets, &t, &f, ty)
                } else {
                    self.pick(out, *cond, &t, &f)
                }
            }
            Expr::Extract(x, k) => {
                let a = self.of(out, *x);
                let lo = self.emit(out, ty, Expr::Extract(a.lo, *k));
                let hi = self.emit(out, ty, Expr::Extract(a.hi, *k));
                let j_lo: Vec<_> =
                    (0..self.n).map(|i| self.emit(out, ty, Expr::Extract(a.j_lo[i], *k))).collect();
                let j_hi: Vec<_> =
                    (0..self.n).map(|i| self.emit(out, ty, Expr::Extract(a.j_hi[i], *k))).collect();
                let lip = if is_float(&self.m.types, ty) {
                    self.tighten(out, a.lip, &j_lo, &j_hi, ty)
                } else {
                    a.lip
                };
                LVal { lo, hi, j_lo, j_hi, lip }
            }
            Expr::Swizzle(x, sw) => {
                let a = self.of(out, *x);
                let lo = self.emit(out, ty, Expr::Swizzle(a.lo, sw.clone()));
                let hi = self.emit(out, ty, Expr::Swizzle(a.hi, sw.clone()));
                let j_lo: Vec<_> = (0..self.n)
                    .map(|i| self.emit(out, ty, Expr::Swizzle(a.j_lo[i], sw.clone())))
                    .collect();
                let j_hi: Vec<_> = (0..self.n)
                    .map(|i| self.emit(out, ty, Expr::Swizzle(a.j_hi[i], sw.clone())))
                    .collect();
                let distinct = (0..sw.len()).all(|i| !sw[i + 1..].contains(&sw[i]));
                let base_lip = if distinct {
                    a.lip
                } else {
                    let sk = self.kf(out, (sw.len() as f64).sqrt());
                    let s = self.bin(out, BinOp::Mul, a.lip, sk, self.f32_ty);
                    self.pad_up(out, s, 4.0)
                };
                let lip = self.tighten(out, base_lip, &j_lo, &j_hi, ty);
                LVal { lo, hi, j_lo, j_hi, lip }
            }
            Expr::Construct(t, comps) if floaty(&self.m.types, *t) => {
                let cs: Vec<LVal> = comps.iter().map(|x| self.of(out, *x)).collect();
                let lo = self.emit(out, ty, Expr::Construct(*t, cs.iter().map(|a| a.lo).collect()));
                let hi = self.emit(out, ty, Expr::Construct(*t, cs.iter().map(|a| a.hi).collect()));
                let j_lo: Vec<_> = (0..self.n)
                    .map(|i| {
                        self.emit(
                            out,
                            ty,
                            Expr::Construct(*t, cs.iter().map(|a| a.j_lo[i]).collect()),
                        )
                    })
                    .collect();
                let j_hi: Vec<_> = (0..self.n)
                    .map(|i| {
                        self.emit(
                            out,
                            ty,
                            Expr::Construct(*t, cs.iter().map(|a| a.j_hi[i]).collect()),
                        )
                    })
                    .collect();
                let lip = if matches!(self.m.types.get(*t), TypeDef::Vector(..)) {
                    let mut sum2 = self.kf(out, 0.0);
                    let cap = self.kf(out, CAP);
                    let mut ok = self.bconst(out, true);
                    for c in &cs {
                        let okc = self.cmp(out, BinOp::Le, c.lip, cap);
                        ok = self.emit(out, self.bool_ty, Expr::Binary(BinOp::And, ok, okc));
                        let lc = self.builtin(out, Builtin::Min, vec![c.lip, cap], self.f32_ty);
                        let sq = self.bin(out, BinOp::Mul, lc, lc, self.f32_ty);
                        sum2 = self.bin(out, BinOp::Add, sum2, sq, self.f32_ty);
                    }
                    let frob = self.builtin(out, Builtin::Sqrt, vec![sum2], self.f32_ty);
                    let frob = self.pad_up(out, frob, 8.0);
                    let big = self.kf(out, BIG);
                    let frob = self.sel(out, ok, frob, big);
                    let row_lips: Vec<ValueId> = cs.iter().map(|c| c.lip).collect();
                    let j_nrm = self.op_norm_hi_rows(out, &j_lo, &j_hi, ty, Some(&row_lips));
                    let frob = self.cap_lip(out, frob);
                    self.builtin(out, Builtin::Min, vec![frob, j_nrm], self.f32_ty)
                } else {
                    let mut m = self.kf(out, 0.0);
                    for c in &cs {
                        m = self.builtin(out, Builtin::Max, vec![m, c.lip], self.f32_ty);
                    }
                    m
                };
                LVal { lo, hi, j_lo, j_hi, lip }
            }
            Expr::Splat(x, k) => {
                let a = self.of(out, *x);
                let lo = self.emit(out, ty, Expr::Splat(a.lo, *k));
                let hi = self.emit(out, ty, Expr::Splat(a.hi, *k));
                let j_lo: Vec<_> =
                    (0..self.n).map(|i| self.emit(out, ty, Expr::Splat(a.j_lo[i], *k))).collect();
                let j_hi: Vec<_> =
                    (0..self.n).map(|i| self.emit(out, ty, Expr::Splat(a.j_hi[i], *k))).collect();
                let sk = self.kf(out, (*k as f64).sqrt());
                let lip = self.bin(out, BinOp::Mul, a.lip, sk, self.f32_ty);
                let lip = self.pad_up(out, lip, 4.0);
                LVal { lo, hi, j_lo, j_hi, lip }
            }
            Expr::Builtin(b, args) => self.builtin_rule(out, *b, args, ty)?,
            _ => self.every(out, ty),
        };
        self.vals.insert(v, a);
        Ok(())
    }

    fn let_bool(&mut self, e: &Expr, out: &mut Block) -> R<Sure> {
        let bt = self.bool_ty;
        Ok(match e {
            Expr::Binary(
                op @ (BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne),
                a,
                b,
            ) if is_float(&self.m.types, self.f.value_ty(*a)) => {
                let ta = self.f.value_ty(*a);
                let tb = self.f.value_ty(*b);
                let ty = match self.m.types.get(ta) {
                    TypeDef::Vector(..) => ta,
                    _ => tb,
                };
                if !matches!(self.m.types.get(ty), TypeDef::Scalar(_)) {
                    return Err(Error::not_derivable("vector comparison in Lipschitz"));
                }
                let aa = self.of_as(out, *a, ty);
                let bb = self.of_as(out, *b, ty);
                let (dlo, dhi) = self.iv_sub(out, (aa.lo, aa.hi), (bb.lo, bb.hi), ty);
                let z = self.kf(out, 0.0);
                match op {
                    BinOp::Lt => {
                        (self.cmp(out, BinOp::Lt, dhi, z), self.cmp(out, BinOp::Ge, dlo, z))
                    }
                    BinOp::Le => {
                        (self.cmp(out, BinOp::Le, dhi, z), self.cmp(out, BinOp::Gt, dlo, z))
                    }
                    BinOp::Gt => {
                        (self.cmp(out, BinOp::Gt, dlo, z), self.cmp(out, BinOp::Le, dhi, z))
                    }
                    BinOp::Ge => {
                        (self.cmp(out, BinOp::Ge, dlo, z), self.cmp(out, BinOp::Lt, dhi, z))
                    }
                    BinOp::Eq | BinOp::Ne => {
                        let above = self.cmp(out, BinOp::Gt, dlo, z);
                        let below = self.cmp(out, BinOp::Lt, dhi, z);
                        let apart = self.emit(out, bt, Expr::Binary(BinOp::Or, above, below));
                        let never = self.bconst(out, false);
                        if *op == BinOp::Eq { (never, apart) } else { (apart, never) }
                    }
                    _ => unreachable!(),
                }
            }
            Expr::Binary(op @ (BinOp::And | BinOp::Or), a, b) => {
                let (x, y) = (self.sure_of(out, *a), self.sure_of(out, *b));
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
            Expr::Param(i) if !self.pparts[*i as usize].is_empty() => {
                let p0 = self.pparts[*i as usize][0];
                let p1 = self.pparts[*i as usize][1];
                (self.emit(out, bt, Expr::Param(p0)), self.emit(out, bt, Expr::Param(p1)))
            }
            Expr::Select { cond, if_true, if_false } => {
                let (t, f) = (self.sure_of(out, *if_true), self.sure_of(out, *if_false));
                if self.act.value(*cond) {
                    let s = self.sure_of(out, *cond);
                    self.join_bool(out, s, t, f)
                } else {
                    (self.sel(out, *cond, t.0, f.0), self.sel(out, *cond, t.1, f.1))
                }
            }
            _ => (self.bconst(out, false), self.bconst(out, false)),
        })
    }

    fn mul_expr(&mut self, out: &mut Block, a: ValueId, b: ValueId, ty: TypeId) -> R<LVal> {
        let (ta, tb) = (self.f.value_ty(a), self.f.value_ty(b));
        let mat = match (self.m.types.get(ta), self.m.types.get(tb)) {
            (TypeDef::Matrix(..), _) => Some((a, b, true)),
            (_, TypeDef::Matrix(..)) => Some((b, a, false)),
            _ => None,
        };
        if let Some((mv, v, left)) = mat {
            if self.act.value(mv) {
                return Err(Error::not_derivable("matrix depending on input in Lipschitz"));
            }
            return Ok(self.mat_mul(out, mv, v, left, ty));
        }
        let aa = self.of_as(out, a, ty);
        let bb = self.of_as(out, b, ty);
        let (lo, hi) = self.iv_mul(out, (aa.lo, aa.hi), (bb.lo, bb.hi), ty);
        let mut j_lo = Vec::with_capacity(self.n);
        let mut j_hi = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let p1 = self.iv_mul(out, (bb.lo, bb.hi), (aa.j_lo[i], aa.j_hi[i]), ty);
            let p2 = self.iv_mul(out, (aa.lo, aa.hi), (bb.j_lo[i], bb.j_hi[i]), ty);
            let (jl, jh) = self.iv_add(out, p1, p2, ty);
            j_lo.push(jl);
            j_hi.push(jh);
        }
        let ma = self.max_abs_scalar(out, (aa.lo, aa.hi), ty);
        let mb = self.max_abs_scalar(out, (bb.lo, bb.hi), ty);
        let t1 = self.bin(out, BinOp::Mul, mb, aa.lip, self.f32_ty);
        let t2 = self.bin(out, BinOp::Mul, ma, bb.lip, self.f32_ty);
        let raw = self.bin(out, BinOp::Add, t1, t2, self.f32_ty);
        let raw = self.pad_up(out, raw, 4.0);
        let lip = self.tighten(out, raw, &j_lo, &j_hi, ty);
        Ok(LVal { lo, hi, j_lo, j_hi, lip })
    }

    fn mat_mul(
        &mut self,
        out: &mut Block,
        mv: ValueId,
        v: ValueId,
        left: bool,
        ty: TypeId,
    ) -> LVal {
        let mt = self.f.value_ty(mv);
        let (cols, rows) = match self.m.types.get(mt) {
            &TypeDef::Matrix(c, r) => (c, r),
            _ => unreachable!("matrix"),
        };
        let col_ty = self.m.types.intern(TypeDef::Vector(Scalar::F32, rows));
        let a = self.of(out, v);
        let vt = self.ty(a.lo);
        let mut abs_cols = Vec::new();
        for j in 0..cols {
            let col = self.emit(out, col_ty, Expr::Extract(mv, j as u32));
            abs_cols.push(self.builtin(out, Builtin::Abs, vec![col], col_ty));
        }
        let am = self.emit(out, mt, Expr::Construct(mt, abs_cols));
        let mm = |s: &mut Self, out: &mut Block, m: ValueId, x: ValueId| {
            let e =
                if left { Expr::Binary(BinOp::Mul, m, x) } else { Expr::Binary(BinOp::Mul, x, m) };
            s.emit(out, ty, e)
        };
        let iv_mm =
            |s: &mut Self, out: &mut Block, (l, h): (ValueId, ValueId)| -> (ValueId, ValueId) {
                let half = s.k(out, vt, 0.5);
                let hl = s.bin(out, BinOp::Mul, l, half, vt);
                let hh = s.bin(out, BinOp::Mul, h, half, vt);
                let mid = s.bin(out, BinOp::Add, hl, hh, vt);
                let rad = s.bin(out, BinOp::Sub, hh, hl, vt);
                let cm = mm(s, out, mv, mid);
                let cr = mm(s, out, am, rad);
                let rlo = s.bin(out, BinOp::Sub, cm, cr, ty);
                let rhi = s.bin(out, BinOp::Add, cm, cr, ty);
                s.pad_iv(out, rlo, rhi, ty, 6.0)
            };
        let (lo, hi) = iv_mm(self, out, (a.lo, a.hi));
        let mut j_lo = Vec::with_capacity(self.n);
        let mut j_hi = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let (jl, jh) = iv_mm(self, out, (a.j_lo[i], a.j_hi[i]));
            j_lo.push(jl);
            j_hi.push(jh);
        }
        let big = self.kf(out, BIG);
        let lip = self.tighten(out, big, &j_lo, &j_hi, ty);
        LVal { lo, hi, j_lo, j_hi, lip }
    }

    fn div_expr(&mut self, out: &mut Block, a: ValueId, b: ValueId, ty: TypeId) -> LVal {
        let aa = self.of_as(out, a, ty);
        if !self.act.value(b) {
            let k = self.coerce(out, b, ty);
            let z = self.zero(out, ty);
            let pos = self.cmp(out, BinOp::Gt, k, z);
            let neg = self.cmp(out, BinOp::Lt, k, z);
            let bt = self.bool_of(ty);
            let nz = self.emit(out, bt, Expr::Binary(BinOp::Or, pos, neg));
            let one = self.k(out, ty, 1.0);
            let ks = self.sel(out, nz, k, one);
            let div_iv = |s: &mut Self, out: &mut Block, (l, h): (ValueId, ValueId)| {
                let d0 = s.bin(out, BinOp::Div, l, ks, ty);
                let d1 = s.bin(out, BinOp::Div, h, ks, ty);
                let rl = s.builtin(out, Builtin::Min, vec![d0, d1], ty);
                let rh = s.builtin(out, Builtin::Max, vec![d0, d1], ty);
                let (rl, rh) = s.pad_iv(out, rl, rh, ty, 4.0);
                let big = s.k(out, ty, BIG);
                let nbig = s.k(out, ty, -BIG);
                (s.sel(out, nz, rl, nbig), s.sel(out, nz, rh, big))
            };
            let (lo, hi) = div_iv(self, out, (aa.lo, aa.hi));
            let mut j_lo = Vec::with_capacity(self.n);
            let mut j_hi = Vec::with_capacity(self.n);
            for i in 0..self.n {
                let (jl, jh) = div_iv(self, out, (aa.j_lo[i], aa.j_hi[i]));
                j_lo.push(jl);
                j_hi.push(jh);
            }
            let ak = self.builtin(out, Builtin::Abs, vec![k], ty);
            let min_k = match self.width(ty) {
                None => ak,
                Some(w) => {
                    let mut m = self.emit(out, self.f32_ty, Expr::Extract(ak, 0));
                    for j in 1..w as u32 {
                        let c = self.emit(out, self.f32_ty, Expr::Extract(ak, j));
                        m = self.builtin(out, Builtin::Min, vec![m, c], self.f32_ty);
                    }
                    m
                }
            };
            let zf = self.kf(out, 0.0);
            let ok = self.cmp(out, BinOp::Gt, min_k, zf);
            let onef = self.kf(out, 1.0);
            let ms = self.sel(out, ok, min_k, onef);
            let raw = self.bin(out, BinOp::Div, aa.lip, ms, self.f32_ty);
            let raw = self.pad_up(out, raw, 4.0);
            let bigf = self.kf(out, BIG);
            let raw = self.sel(out, ok, raw, bigf);
            let lip = self.tighten(out, raw, &j_lo, &j_hi, ty);
            return LVal { lo, hi, j_lo, j_hi, lip };
        }
        let bb = self.of_as(out, b, ty);
        let z = self.zero(out, ty);
        let pos = self.cmp(out, BinOp::Gt, bb.lo, z);
        let neg = self.cmp(out, BinOp::Lt, bb.hi, z);
        let bt = self.bool_of(ty);
        let ok = self.emit(out, bt, Expr::Binary(BinOp::Or, pos, neg));
        let one = self.k(out, ty, 1.0);
        let bl = self.sel(out, ok, bb.lo, one);
        let bh = self.sel(out, ok, bb.hi, one);
        let rl = self.bin(out, BinOp::Div, one, bh, ty);
        let rh = self.bin(out, BinOp::Div, one, bl, ty);
        let (rl, rh) = self.pad_iv(out, rl, rh, ty, 4.0);
        let (qlo, qhi) = self.iv_mul(out, (aa.lo, aa.hi), (rl, rh), ty);
        let big = self.k(out, ty, BIG);
        let nbig = self.k(out, ty, -BIG);
        let lo = self.sel(out, ok, qlo, nbig);
        let hi = self.sel(out, ok, qhi, big);
        let mut j_lo = Vec::with_capacity(self.n);
        let mut j_hi = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let qb = self.iv_mul(out, (lo, hi), (bb.j_lo[i], bb.j_hi[i]), ty);
            let num = self.iv_sub(out, (aa.j_lo[i], aa.j_hi[i]), qb, ty);
            let (jl, jh) = self.iv_mul(out, num, (rl, rh), ty);
            j_lo.push(self.sel(out, ok, jl, nbig));
            j_hi.push(self.sel(out, ok, jh, big));
        }
        let (bmin_v, _) = self.iv_abs(out, (bb.lo, bb.hi), ty);
        let bmin = match self.width(ty) {
            None => bmin_v,
            Some(w) => {
                let mut m = self.emit(out, self.f32_ty, Expr::Extract(bmin_v, 0));
                for j in 1..w as u32 {
                    let c = self.emit(out, self.f32_ty, Expr::Extract(bmin_v, j));
                    m = self.builtin(out, Builtin::Min, vec![m, c], self.f32_ty);
                }
                m
            }
        };
        let zf = self.kf(out, 0.0);
        let okf = self.cmp(out, BinOp::Gt, bmin, zf);
        let onef = self.kf(out, 1.0);
        let bs = self.sel(out, okf, bmin, onef);
        let mq = self.max_abs_scalar(out, (lo, hi), ty);
        let qb_l = self.bin(out, BinOp::Mul, mq, bb.lip, self.f32_ty);
        let num_l = self.bin(out, BinOp::Add, aa.lip, qb_l, self.f32_ty);
        let raw = self.bin(out, BinOp::Div, num_l, bs, self.f32_ty);
        let raw = self.pad_up(out, raw, 6.0);
        let bigf = self.kf(out, BIG);
        let raw = self.sel(out, okf, raw, bigf);
        let lip = self.tighten(out, raw, &j_lo, &j_hi, ty);
        LVal { lo, hi, j_lo, j_hi, lip }
    }

    /// Chain-rule helper for a component-wise smooth unary op with value interval `(lo, hi)` and
    /// derivative interval `(dlo, dhi)`.
    fn chain_unary(
        &mut self,
        out: &mut Block,
        a: &LVal,
        val: (ValueId, ValueId),
        deriv: (ValueId, ValueId),
        ty: TypeId,
    ) -> LVal {
        let mut j_lo = Vec::with_capacity(self.n);
        let mut j_hi = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let (jl, jh) = self.iv_mul(out, deriv, (a.j_lo[i], a.j_hi[i]), ty);
            j_lo.push(jl);
            j_hi.push(jh);
        }
        let md = self.max_abs_scalar(out, deriv, ty);
        let raw = self.bin(out, BinOp::Mul, md, a.lip, self.f32_ty);
        let raw = self.pad_up(out, raw, 4.0);
        let lip = self.tighten(out, raw, &j_lo, &j_hi, ty);
        LVal { lo: val.0, hi: val.1, j_lo, j_hi, lip }
    }

    fn min_max_val(
        &mut self,
        out: &mut Block,
        a: &LVal,
        b: &LVal,
        b_inactive: bool,
        ty: TypeId,
        is_max: bool,
    ) -> LVal {
        let op = if is_max { Builtin::Max } else { Builtin::Min };
        let lo = self.builtin(out, op, vec![a.lo, b.lo], ty);
        let hi = self.builtin(out, op, vec![a.hi, b.hi], ty);
        let win_a = if is_max {
            self.cmp(out, BinOp::Ge, a.lo, b.hi)
        } else {
            self.cmp(out, BinOp::Le, a.hi, b.lo)
        };
        let win_b = if is_max {
            self.cmp(out, BinOp::Ge, b.lo, a.hi)
        } else {
            self.cmp(out, BinOp::Le, b.hi, a.lo)
        };
        let mut j_lo = Vec::with_capacity(self.n);
        let mut j_hi = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let (hl, hh) = self.iv_hull(out, (a.j_lo[i], a.j_hi[i]), (b.j_lo[i], b.j_hi[i]), ty);
            let jl = self.sel(out, win_b, b.j_lo[i], hl);
            let jh = self.sel(out, win_b, b.j_hi[i], hh);
            let jl = self.sel(out, win_a, a.j_lo[i], jl);
            let jh = self.sel(out, win_a, a.j_hi[i], jh);
            j_lo.push(jl);
            j_hi.push(jh);
        }
        let lip = if self.width(ty).is_none() {
            let both = self.builtin(out, Builtin::Max, vec![a.lip, b.lip], self.f32_ty);
            let r = self.sel(out, win_b, b.lip, both);
            let r = self.sel(out, win_a, a.lip, r);
            self.tighten(out, r, &j_lo, &j_hi, ty)
        } else if b_inactive {
            // Diagonal scaling theorem: `max(v, const)` has `J = D J_v` with `D_ii ∈ {0, 1}`.
            self.tighten(out, a.lip, &j_lo, &j_hi, ty)
        } else {
            // Disjoint orthogonal diagonal projections: `||D J_a + (I-D) J_b||₂ <= √(L_a² + L_b²)`.
            let cap = self.kf(out, CAP);
            let la = self.builtin(out, Builtin::Min, vec![a.lip, cap], self.f32_ty);
            let lb = self.builtin(out, Builtin::Min, vec![b.lip, cap], self.f32_ty);
            let a2 = self.bin(out, BinOp::Mul, la, la, self.f32_ty);
            let b2 = self.bin(out, BinOp::Mul, lb, lb, self.f32_ty);
            let s2 = self.bin(out, BinOp::Add, a2, b2, self.f32_ty);
            let comb = self.builtin(out, Builtin::Sqrt, vec![s2], self.f32_ty);
            let comb = self.pad_up(out, comb, 4.0);
            self.tighten(out, comb, &j_lo, &j_hi, ty)
        };
        LVal { lo, hi, j_lo, j_hi, lip }
    }

    fn builtin_rule(
        &mut self,
        out: &mut Block,
        b: Builtin,
        args: &[ValueId],
        ty: TypeId,
    ) -> R<LVal> {
        use Builtin as B;
        let f = self.f32_ty;
        Ok(match b {
            B::Abs => {
                let a = self.of_as(out, args[0], ty);
                let (lo, hi) = self.iv_abs(out, (a.lo, a.hi), ty);
                let z = self.zero(out, ty);
                let pos = self.cmp(out, BinOp::Ge, a.lo, z);
                let neg = self.cmp(out, BinOp::Le, a.hi, z);
                let mut j_lo = Vec::with_capacity(self.n);
                let mut j_hi = Vec::with_capacity(self.n);
                for i in 0..self.n {
                    let (nl, nh) = self.iv_neg(out, (a.j_lo[i], a.j_hi[i]), ty);
                    let (hl, hh) = self.iv_hull(out, (a.j_lo[i], a.j_hi[i]), (nl, nh), ty);
                    let jl = self.sel(out, neg, nl, hl);
                    let jh = self.sel(out, neg, nh, hh);
                    j_lo.push(self.sel(out, pos, a.j_lo[i], jl));
                    j_hi.push(self.sel(out, pos, a.j_hi[i], jh));
                }
                // Diagonal scaling theorem: `||S J_a||₂ <= ||J_a||₂` identically!
                LVal { lo, hi, j_lo, j_hi, lip: a.lip }
            }
            B::Min | B::Max => {
                let a_in = !self.act.value(args[0]);
                let b_in = !self.act.value(args[1]);
                let a = self.of_as(out, args[0], ty);
                let c = self.of_as(out, args[1], ty);
                if a_in {
                    self.min_max_val(out, &c, &a, true, ty, b == B::Max)
                } else {
                    self.min_max_val(out, &a, &c, b_in, ty, b == B::Max)
                }
            }
            B::Clamp => {
                let a = self.of_as(out, args[0], ty);
                let lo = self.of_as(out, args[1], ty);
                let hi = self.of_as(out, args[2], ty);
                let m = self.min_max_val(out, &a, &lo, !self.act.value(args[1]), ty, true);
                self.min_max_val(out, &m, &hi, !self.act.value(args[2]), ty, false)
            }
            B::Saturate => {
                let a = self.of_as(out, args[0], ty);
                let z = self.zero(out, ty);
                let one = self.k(out, ty, 1.0);
                let z = self.exact(out, z);
                let one = self.exact(out, one);
                let m = self.min_max_val(out, &a, &z, true, ty, true);
                self.min_max_val(out, &m, &one, true, ty, false)
            }
            B::Length | B::Distance => {
                let at = self.f.value_ty(args[0]);
                let mut a = self.of_as(out, args[0], at);
                if b == B::Distance {
                    let c = self.of_as(out, args[1], at);
                    let (lo, hi) = self.iv_sub(out, (a.lo, a.hi), (c.lo, c.hi), at);
                    let mut j_lo = Vec::with_capacity(self.n);
                    let mut j_hi = Vec::with_capacity(self.n);
                    for i in 0..self.n {
                        let (jl, jh) =
                            self.iv_sub(out, (a.j_lo[i], a.j_hi[i]), (c.j_lo[i], c.j_hi[i]), at);
                        j_lo.push(jl);
                        j_hi.push(jh);
                    }
                    let sl = self.bin(out, BinOp::Add, a.lip, c.lip, f);
                    let sl = self.pad_up(out, sl, 2.0);
                    let lip = self.tighten(out, sl, &j_lo, &j_hi, at);
                    a = LVal { lo, hi, j_lo, j_hi, lip };
                }
                let lo = self.vec_len_lo(out, (a.lo, a.hi), at);
                let hi = self.vec_len_hi(out, (a.lo, a.hi), at);
                let zf = self.kf(out, 0.0);
                let safe = self.cmp(out, BinOp::Gt, lo, zf);
                let onef = self.kf(out, 1.0);
                let lo_s = self.sel(out, safe, lo, onef);
                let hi_s = self.sel(out, safe, hi, onef);
                let inv_lo = self.bin(out, BinOp::Div, onef, hi_s, f);
                let inv_hi = self.bin(out, BinOp::Div, onef, lo_s, f);
                let inv_lo = self.coerce(out, inv_lo, at);
                let inv_hi = self.coerce(out, inv_hi, at);
                let (ulo, uhi) = self.iv_mul(out, (a.lo, a.hi), (inv_lo, inv_hi), at);
                let one_v = self.k(out, at, 1.0);
                let none_v = self.k(out, at, -1.0);
                let ulo = self.builtin(out, Builtin::Clamp, vec![ulo, none_v, one_v], at);
                let uhi = self.builtin(out, Builtin::Clamp, vec![uhi, ulo, one_v], at);
                let mut j_lo = Vec::with_capacity(self.n);
                let mut j_hi = Vec::with_capacity(self.n);
                for i in 0..self.n {
                    let p = self.iv_mul(out, (ulo, uhi), (a.j_lo[i], a.j_hi[i]), at);
                    let (dl, dh) = self.iv_sum_comps(out, p, at);
                    // Per-axis Cauchy-Schwarz: `|∂||a||/∂x_i| <= min(a.lip, ||∂a/∂x_i||₂)`,
                    // preserving exact zero partial derivatives (e.g. `∂||p.xz||/∂y == 0` on axis!).
                    let col_nrm = self.vec_len_hi(out, (a.j_lo[i], a.j_hi[i]), at);
                    let bound_i = self.builtin(out, Builtin::Min, vec![a.lip, col_nrm], f);
                    let nbound_i = self.emit(out, f, Expr::Unary(UnOp::Neg, bound_i));
                    j_lo.push(self.sel(out, safe, dl, nbound_i));
                    j_hi.push(self.sel(out, safe, dh, bound_i));
                }
                // Cauchy-Schwarz: `||(v/|v|)ᵀ J_v||₂ <= ||J_v||₂ <= a.lip` EVERYWHERE, even at 0!
                let lip = self.tighten(out, a.lip, &j_lo, &j_hi, f);
                LVal { lo, hi, j_lo, j_hi, lip }
            }
            B::Dot => {
                let at = self.f.value_ty(args[0]);
                let a = self.of_as(out, args[0], at);
                let c = self.of_as(out, args[1], at);
                let p = self.iv_mul(out, (a.lo, a.hi), (c.lo, c.hi), at);
                let (lo, hi) = self.iv_sum_comps(out, p, at);
                let mut j_lo = Vec::with_capacity(self.n);
                let mut j_hi = Vec::with_capacity(self.n);
                for i in 0..self.n {
                    let p1 = self.iv_mul(out, (c.lo, c.hi), (a.j_lo[i], a.j_hi[i]), at);
                    let p2 = self.iv_mul(out, (a.lo, a.hi), (c.j_lo[i], c.j_hi[i]), at);
                    let s1 = self.iv_sum_comps(out, p1, at);
                    let s2 = self.iv_sum_comps(out, p2, at);
                    let (jl, jh) = self.iv_add(out, s1, s2, f);
                    j_lo.push(jl);
                    j_hi.push(jh);
                }
                let la = self.vec_len_hi(out, (a.lo, a.hi), at);
                let lc = self.vec_len_hi(out, (c.lo, c.hi), at);
                let t1 = self.bin(out, BinOp::Mul, lc, a.lip, f);
                let t2 = self.bin(out, BinOp::Mul, la, c.lip, f);
                let raw = self.bin(out, BinOp::Add, t1, t2, f);
                let raw = self.pad_up(out, raw, 6.0);
                let lip = self.tighten(out, raw, &j_lo, &j_hi, f);
                LVal { lo, hi, j_lo, j_hi, lip }
            }
            B::Sqrt => {
                let a = self.of_as(out, args[0], ty);
                let z = self.zero(out, ty);
                let lo0 = self.builtin(out, Builtin::Max, vec![a.lo, z], ty);
                let hi0 = self.builtin(out, Builtin::Max, vec![a.hi, lo0], ty);
                let slo = self.builtin(out, Builtin::Sqrt, vec![lo0], ty);
                let shi = self.builtin(out, Builtin::Sqrt, vec![hi0], ty);
                let (vlo, vhi) = self.pad_iv(out, slo, shi, ty, 4.0);
                let vlo = self.builtin(out, Builtin::Max, vec![vlo, z], ty);
                let pos = self.cmp(out, BinOp::Gt, lo0, z);
                let one = self.k(out, ty, 1.0);
                let half = self.k(out, ty, 0.5);
                let sl_s = self.sel(out, pos, slo, one);
                let sh_s = self.sel(out, pos, shi, one);
                let dlo = self.bin(out, BinOp::Div, half, sh_s, ty);
                let dhi = self.bin(out, BinOp::Div, half, sl_s, ty);
                let (dlo, dhi) = self.pad_iv(out, dlo, dhi, ty, 4.0);
                let big = self.k(out, ty, BIG);
                let dlo = self.sel(out, pos, dlo, z);
                let dhi = self.sel(out, pos, dhi, big);
                self.chain_unary(out, &a, (vlo, vhi), (dlo, dhi), ty)
            }
            B::Sin | B::Cos => {
                let a = self.of_as(out, args[0], ty);
                let is_cos = b == B::Cos;
                let (vlo, vhi) = self.iv_trig(out, (a.lo, a.hi), ty, is_cos);
                let (dlo, dhi) = if is_cos {
                    let (sl, sh) = self.iv_trig(out, (a.lo, a.hi), ty, false);
                    self.iv_neg(out, (sl, sh), ty)
                } else {
                    self.iv_trig(out, (a.lo, a.hi), ty, true)
                };
                self.chain_unary(out, &a, (vlo, vhi), (dlo, dhi), ty)
            }
            B::Exp | B::Exp2 => {
                let a = self.of_as(out, args[0], ty);
                let ceil = self.k(out, ty, if b == B::Exp { 80.0 } else { 115.0 });
                let floor = self.k(out, ty, -100.0);
                let lo = self.builtin(out, Builtin::Clamp, vec![a.lo, floor, ceil], ty);
                let hi = self.builtin(out, Builtin::Clamp, vec![a.hi, lo, ceil], ty);
                let fl = self.builtin(out, b, vec![lo], ty);
                let fh = self.builtin(out, b, vec![hi], ty);
                let (vlo, vhi) = self.pad_iv(out, fl, fh, ty, 8.0);
                let z = self.zero(out, ty);
                let vlo = self.builtin(out, Builtin::Max, vec![vlo, z], ty);
                let fits = self.cmp(out, BinOp::Le, a.hi, ceil);
                let big = self.k(out, ty, BIG);
                let vhi = self.sel(out, fits, vhi, big);
                let deriv = if b == B::Exp {
                    (vlo, vhi)
                } else {
                    let ln2 = self.k(out, ty, std::f64::consts::LN_2);
                    self.iv_mul(out, (vlo, vhi), (ln2, ln2), ty)
                };
                self.chain_unary(out, &a, (vlo, vhi), deriv, ty)
            }
            B::Log | B::Log2 => {
                let a = self.of_as(out, args[0], ty);
                let z = self.zero(out, ty);
                let ok = self.cmp(out, BinOp::Gt, a.lo, z);
                let one = self.k(out, ty, 1.0);
                let ls = self.sel(out, ok, a.lo, one);
                let hs = self.sel(out, ok, a.hi, one);
                let fl = self.builtin(out, b, vec![ls], ty);
                let fh = self.builtin(out, b, vec![hs], ty);
                let (vlo, vhi) = self.pad_iv(out, fl, fh, ty, 8.0);
                let big = self.k(out, ty, BIG);
                let nbig = self.k(out, ty, -BIG);
                let vlo = self.sel(out, ok, vlo, nbig);
                let vhi = self.sel(out, ok, vhi, big);
                let scale =
                    self.k(out, ty, if b == B::Log2 { std::f64::consts::LOG2_E } else { 1.0 });
                let dlo = self.bin(out, BinOp::Div, scale, hs, ty);
                let dhi = self.bin(out, BinOp::Div, scale, ls, ty);
                let (dlo, dhi) = self.pad_iv(out, dlo, dhi, ty, 4.0);
                let dlo = self.sel(out, ok, dlo, z);
                let dhi = self.sel(out, ok, dhi, big);
                self.chain_unary(out, &a, (vlo, vhi), (dlo, dhi), ty)
            }
            B::Atan => {
                let a = self.of_as(out, args[0], ty);
                let fl = self.builtin(out, Builtin::Atan, vec![a.lo], ty);
                let fh = self.builtin(out, Builtin::Atan, vec![a.hi], ty);
                let (vlo, vhi) = self.pad_iv(out, fl, fh, ty, 8.0);
                let (amin, amax) = self.iv_abs(out, (a.lo, a.hi), ty);
                let one = self.k(out, ty, 1.0);
                let cap = self.k(out, ty, CAP);
                let amin = self.builtin(out, Builtin::Min, vec![amin, cap], ty);
                let amax = self.builtin(out, Builtin::Min, vec![amax, cap], ty);
                let amin2 = self.bin(out, BinOp::Mul, amin, amin, ty);
                let amax2 = self.bin(out, BinOp::Mul, amax, amax, ty);
                let den_lo = self.bin(out, BinOp::Add, one, amin2, ty);
                let den_hi = self.bin(out, BinOp::Add, one, amax2, ty);
                let dlo = self.bin(out, BinOp::Div, one, den_hi, ty);
                let dhi = self.bin(out, BinOp::Div, one, den_lo, ty);
                let (dlo, dhi) = self.pad_iv(out, dlo, dhi, ty, 4.0);
                self.chain_unary(out, &a, (vlo, vhi), (dlo, dhi), ty)
            }
            B::Smoothstep if !self.act.value(args[0]) && !self.act.value(args[1]) => {
                let x = self.of_as(out, args[2], ty);
                let e0 = self.coerce(out, args[0], ty);
                let e1 = self.coerce(out, args[1], ty);
                let w = self.bin(out, BinOp::Sub, e1, e0, ty);
                let z = self.zero(out, ty);
                let one = self.k(out, ty, 1.0);
                // `t = clamp((x - e0) / (e1 - e0), 0, 1)`
                let (dlo, dhi) = self.iv_sub(out, (x.lo, x.hi), (e0, e0), ty);
                let t0 = self.bin(out, BinOp::Div, dlo, w, ty);
                let t1 = self.bin(out, BinOp::Div, dhi, w, ty);
                let tmin = self.builtin(out, Builtin::Min, vec![t0, t1], ty);
                let tmax = self.builtin(out, Builtin::Max, vec![t0, t1], ty);
                let tlo = self.builtin(out, Builtin::Clamp, vec![tmin, z, one], ty);
                let thi = self.builtin(out, Builtin::Clamp, vec![tmax, z, one], ty);
                // Monotone value `t²(3 - 2t)`:
                let eval_ss = |s: &mut Self, out: &mut Block, t: ValueId| {
                    let t2 = s.bin(out, BinOp::Mul, t, t, ty);
                    let two = s.k(out, ty, 2.0);
                    let three = s.k(out, ty, 3.0);
                    let tt = s.bin(out, BinOp::Mul, two, t, ty);
                    let u = s.bin(out, BinOp::Sub, three, tt, ty);
                    s.bin(out, BinOp::Mul, t2, u, ty)
                };
                let vlo = eval_ss(self, out, tlo);
                let vhi = eval_ss(self, out, thi);
                // Derivative `6 t (1 - t) / (e1 - e0)` (0 if `t` is clamped to 0 or 1!).
                let six = self.k(out, ty, 6.0);
                let half = self.k(out, ty, 0.5);
                let omtl = self.bin(out, BinOp::Sub, one, tlo, ty);
                let tl_omtl = self.bin(out, BinOp::Mul, tlo, omtl, ty);
                let at_lo = self.bin(out, BinOp::Mul, six, tl_omtl, ty);
                let omth = self.bin(out, BinOp::Sub, one, thi, ty);
                let th_omth = self.bin(out, BinOp::Mul, thi, omth, ty);
                let at_hi = self.bin(out, BinOp::Mul, six, th_omth, ty);
                let p_lo = self.builtin(out, Builtin::Min, vec![at_lo, at_hi], ty);
                let p_hi0 = self.builtin(out, Builtin::Max, vec![at_lo, at_hi], ty);
                let bt = self.bool_of(ty);
                let le_h = self.cmp(out, BinOp::Le, tlo, half);
                let ge_h = self.cmp(out, BinOp::Ge, thi, half);
                let spans_half = self.emit(out, bt, Expr::Binary(BinOp::And, le_h, ge_h));
                let peak = self.k(out, ty, 1.5);
                let p_hi = self.sel(out, spans_half, peak, p_hi0);
                // Clamped outside `[0, 1]` -> zero derivative.
                let le_z = self.cmp(out, BinOp::Le, thi, z);
                let ge_o = self.cmp(out, BinOp::Ge, tlo, one);
                let clamped = self.emit(out, bt, Expr::Binary(BinOp::Or, le_z, ge_o));
                let p_lo = self.sel(out, clamped, z, p_lo);
                let p_hi = self.sel(out, clamped, z, p_hi);
                let d0 = self.bin(out, BinOp::Div, p_lo, w, ty);
                let d1 = self.bin(out, BinOp::Div, p_hi, w, ty);
                let dlo = self.builtin(out, Builtin::Min, vec![d0, d1], ty);
                let dhi = self.builtin(out, Builtin::Max, vec![d0, d1], ty);
                let (dlo, dhi) = self.pad_iv(out, dlo, dhi, ty, 8.0);
                self.chain_unary(out, &x, (vlo, vhi), (dlo, dhi), ty)
            }
            B::Mix => {
                let a = self.of_as(out, args[0], ty);
                let c = self.of_as(out, args[1], ty);
                let t = self.of_as(out, args[2], ty);
                let one = self.k(out, ty, 1.0);
                let omt = self.iv_sub(out, (one, one), (t.lo, t.hi), ty);
                let p1 = self.iv_mul(out, (a.lo, a.hi), omt, ty);
                let p2 = self.iv_mul(out, (c.lo, c.hi), (t.lo, t.hi), ty);
                let (lo, hi) = self.iv_add(out, p1, p2, ty);
                let cma = self.iv_sub(out, (c.lo, c.hi), (a.lo, a.hi), ty);
                let mut j_lo = Vec::with_capacity(self.n);
                let mut j_hi = Vec::with_capacity(self.n);
                for i in 0..self.n {
                    let t1 = self.iv_mul(out, omt, (a.j_lo[i], a.j_hi[i]), ty);
                    let t2 = self.iv_mul(out, (t.lo, t.hi), (c.j_lo[i], c.j_hi[i]), ty);
                    let t3 = self.iv_mul(out, cma, (t.j_lo[i], t.j_hi[i]), ty);
                    let s = self.iv_add(out, t1, t2, ty);
                    let (jl, jh) = self.iv_add(out, s, t3, ty);
                    j_lo.push(jl);
                    j_hi.push(jh);
                }
                let momt = self.max_abs_scalar(out, omt, ty);
                let mt = self.max_abs_scalar(out, (t.lo, t.hi), ty);
                let mcma = self.max_abs_scalar(out, cma, ty);
                let l1 = self.bin(out, BinOp::Mul, momt, a.lip, f);
                let l2 = self.bin(out, BinOp::Mul, mt, c.lip, f);
                let l3 = self.bin(out, BinOp::Mul, mcma, t.lip, f);
                let l12 = self.bin(out, BinOp::Add, l1, l2, f);
                let raw = self.bin(out, BinOp::Add, l12, l3, f);
                let raw = self.pad_up(out, raw, 6.0);
                let lip = self.tighten(out, raw, &j_lo, &j_hi, ty);
                LVal { lo, hi, j_lo, j_hi, lip }
            }
            B::Floor | B::Ceil | B::Round | B::Trunc | B::Sign => {
                let a = self.of_as(out, args[0], ty);
                let fl = self.builtin(out, b, vec![a.lo], ty);
                let fh = self.builtin(out, b, vec![a.hi], ty);
                self.stepwise(out, fl, fh, ty)
            }
            B::Step if !self.act.value(args[0]) => {
                let a = self.of_as(out, args[1], ty);
                let edge = self.coerce(out, args[0], ty);
                let fl = self.builtin(out, b, vec![edge, a.lo], ty);
                let fh = self.builtin(out, b, vec![edge, a.hi], ty);
                self.stepwise(out, fl, fh, ty)
            }
            B::Fract => {
                let a = self.of_as(out, args[0], ty);
                let z = self.zero(out, ty);
                let one = self.k(out, ty, 1.0);
                let fl = self.builtin(out, Builtin::Floor, vec![a.lo], ty);
                let fh = self.builtin(out, Builtin::Floor, vec![a.hi], ty);
                let same = self.cmp(out, BinOp::Eq, fl, fh);
                let rlo = self.bin(out, BinOp::Sub, a.lo, fl, ty);
                let rhi = self.bin(out, BinOp::Sub, a.hi, fl, ty);
                let lo = self.sel(out, same, rlo, z);
                let hi = self.sel(out, same, rhi, one);
                // `fract(x) = x - floor(x)`: its argument's slope where the box holds no integer;
                // where it holds one, a jump of 1, and no bound.
                let flat = self.all_of(out, same, ty);
                let big = self.k(out, ty, BIG);
                let nbig = self.k(out, ty, -BIG);
                let j_lo = a.j_lo.iter().map(|&j| self.sel(out, same, j, nbig)).collect();
                let j_hi = a.j_hi.iter().map(|&j| self.sel(out, same, j, big)).collect();
                let bigf = self.kf(out, BIG);
                let lip = self.sel(out, flat, a.lip, bigf);
                LVal { lo, hi, j_lo, j_hi, lip }
            }
            B::Pow if !self.act.value(args[1]) => {
                let a = self.of_as(out, args[0], ty);
                let p = self.coerce(out, args[1], ty);
                let z = self.zero(out, ty);
                let one = self.k(out, ty, 1.0);
                let ok = self.cmp(out, BinOp::Gt, a.lo, z);
                let ls = self.sel(out, ok, a.lo, one);
                let hs = self.sel(out, ok, a.hi, one);
                let fl = self.builtin(out, Builtin::Pow, vec![ls, p], ty);
                let fh = self.builtin(out, Builtin::Pow, vec![hs, p], ty);
                let vlo = self.builtin(out, Builtin::Min, vec![fl, fh], ty);
                let vhi = self.builtin(out, Builtin::Max, vec![fl, fh], ty);
                let (vlo, vhi) = self.pad_iv(out, vlo, vhi, ty, 16.0);
                let pm1 = self.bin(out, BinOp::Sub, p, one, ty);
                let pow_l = self.builtin(out, Builtin::Pow, vec![ls, pm1], ty);
                let pow_h = self.builtin(out, Builtin::Pow, vec![hs, pm1], ty);
                let dl0 = self.bin(out, BinOp::Mul, p, pow_l, ty);
                let dh0 = self.bin(out, BinOp::Mul, p, pow_h, ty);
                let dlo = self.builtin(out, Builtin::Min, vec![dl0, dh0], ty);
                let dhi = self.builtin(out, Builtin::Max, vec![dl0, dh0], ty);
                let (dlo, dhi) = self.pad_iv(out, dlo, dhi, ty, 16.0);
                let big = self.k(out, ty, BIG);
                let nbig = self.k(out, ty, -BIG);
                let vlo = self.sel(out, ok, vlo, nbig);
                let vhi = self.sel(out, ok, vhi, big);
                let dlo = self.sel(out, ok, dlo, nbig);
                let dhi = self.sel(out, ok, dhi, big);
                self.chain_unary(out, &a, (vlo, vhi), (dlo, dhi), ty)
            }
            B::Normalize => {
                let a = self.of_as(out, args[0], ty);
                let lo_l = self.vec_len_lo(out, (a.lo, a.hi), ty);
                let hi_l = self.vec_len_hi(out, (a.lo, a.hi), ty);
                let zf = self.kf(out, 0.0);
                let ok = self.cmp(out, BinOp::Gt, lo_l, zf);
                let onef = self.kf(out, 1.0);
                let ls = self.sel(out, ok, lo_l, onef);
                let hs = self.sel(out, ok, hi_l, onef);
                let ilo_s = self.bin(out, BinOp::Div, onef, hs, f);
                let ihi_s = self.bin(out, BinOp::Div, onef, ls, f);
                let inv_lo = self.coerce(out, ilo_s, ty);
                let inv_hi = self.coerce(out, ihi_s, ty);
                let (vlo, vhi) = self.iv_mul(out, (a.lo, a.hi), (inv_lo, inv_hi), ty);
                let one_v = self.k(out, ty, 1.0);
                let none_v = self.k(out, ty, -1.0);
                let vlo = self.builtin(out, Builtin::Clamp, vec![vlo, none_v, one_v], ty);
                let vhi = self.builtin(out, Builtin::Clamp, vec![vhi, vlo, one_v], ty);
                // `||J_{normalize(a)}||₂ = ||(I - uuᵀ) / ||a|| · J_a||₂ <= a.lip / lo_l`!
                let raw = self.bin(out, BinOp::Div, a.lip, ls, f);
                let raw = self.pad_up(out, raw, 8.0);
                let bigf = self.kf(out, BIG);
                let lip = self.sel(out, ok, raw, bigf);
                let nlip = self.emit(out, f, Expr::Unary(UnOp::Neg, lip));
                let neg_l = self.coerce(out, nlip, ty);
                let pos_l = self.coerce(out, lip, ty);
                LVal { lo: vlo, hi: vhi, j_lo: vec![neg_l; self.n], j_hi: vec![pos_l; self.n], lip }
            }
            B::Cross => {
                let a = self.of_as(out, args[0], ty);
                let c = self.of_as(out, args[1], ty);
                let la = self.vec_len_hi(out, (a.lo, a.hi), ty);
                let lc = self.vec_len_hi(out, (c.lo, c.hi), ty);
                let m = self.bin(out, BinOp::Mul, la, lc, f);
                let m = self.pad_up(out, m, 6.0);
                let nm = self.emit(out, f, Expr::Unary(UnOp::Neg, m));
                let lo = self.coerce(out, nm, ty);
                let hi = self.coerce(out, m, ty);
                let t1 = self.bin(out, BinOp::Mul, lc, a.lip, f);
                let t2 = self.bin(out, BinOp::Mul, la, c.lip, f);
                let t12 = self.bin(out, BinOp::Add, t1, t2, f);
                let lip = self.pad_up(out, t12, 6.0);
                let nlip = self.emit(out, f, Expr::Unary(UnOp::Neg, lip));
                let nl = self.coerce(out, nlip, ty);
                let pl = self.coerce(out, lip, ty);
                LVal { lo, hi, j_lo: vec![nl; self.n], j_hi: vec![pl; self.n], lip }
            }
            _ => self.every(out, ty),
        })
    }

    fn iv_sum_comps(
        &mut self,
        out: &mut Block,
        a: (ValueId, ValueId),
        at: TypeId,
    ) -> (ValueId, ValueId) {
        let Some(k) = self.width(at) else {
            return a;
        };
        let f = self.f32_ty;
        let mut lo = self.emit(out, f, Expr::Extract(a.0, 0));
        let mut hi = self.emit(out, f, Expr::Extract(a.1, 0));
        for j in 1..k as u32 {
            let cl = self.emit(out, f, Expr::Extract(a.0, j));
            let ch = self.emit(out, f, Expr::Extract(a.1, j));
            (lo, hi) = self.iv_add(out, (lo, hi), (cl, ch), f);
        }
        (lo, hi)
    }

    fn iv_trig(
        &mut self,
        out: &mut Block,
        (lo, hi): (ValueId, ValueId),
        ty: TypeId,
        is_cos: bool,
    ) -> (ValueId, ValueId) {
        let b = if is_cos { Builtin::Cos } else { Builtin::Sin };
        let w = self.bin(out, BinOp::Sub, hi, lo, ty);
        let half = self.k(out, ty, 0.5);
        let r = self.bin(out, BinOp::Mul, w, half, ty);
        let mid = self.bin(out, BinOp::Add, lo, r, ty);
        let fm = self.builtin(out, b, vec![mid], ty);
        let rl = self.bin(out, BinOp::Sub, fm, r, ty);
        let rh = self.bin(out, BinOp::Add, fm, r, ty);
        let (rl, rh) = self.pad_iv(out, rl, rh, ty, 8.0);
        let one = self.k(out, ty, 1.0);
        let none = self.k(out, ty, -1.0);
        let rl = self.builtin(out, Builtin::Max, vec![rl, none], ty);
        let rh = self.builtin(out, Builtin::Min, vec![rh, one], ty);
        (rl, rh)
    }

    // ---- Calls, smin, value_noise ----------------------------------------------------------

    fn arg_lval(&mut self, out: &mut Block, a: &Arg) -> R<LVal> {
        match a {
            Arg::Value(x) => Ok(self.of(out, *x)),
            Arg::Place(p) => {
                let ty = place_type(self.m, self.f, p)?;
                if !floaty(&self.m.types, ty) {
                    return Err(Error::not_derivable("active non-float argument in Lipschitz"));
                }
                if !self.active_root(&p.root) {
                    let v = self.emit(out, ty, Expr::Load(p.clone()));
                    return Ok(self.exact(out, v));
                }
                self.load_lval(out, p, ty)
            }
        }
    }

    fn pass(&mut self, out: &mut Block, v: ValueId, by_ref: bool) -> Arg {
        if !by_ref {
            return Arg::Value(v);
        }
        let ty = self.ty(v);
        let l = self.nf.new_local("arg", ty);
        out.push(Stmt::Store(Place::local(l), v));
        Arg::Place(Place::local(l))
    }

    fn call(&mut self, v: Option<ValueId>, g: FuncId, args: &[Arg], out: &mut Block) -> R<()> {
        let mask: Vec<Bits> = args.iter().map(|a| arg_bits(self.m, self.f, &self.act, a)).collect();
        let known = self.m.functions[g.index()].known;
        let active_v = v.is_some_and(|v| self.act.value(v));
        if let Some(v) = v.filter(|_| active_v) {
            if known == Some(Known::ValueNoise) {
                let x = self.arg_lval(out, &args[0])?;
                let a = self.noise(out, g, x)?;
                self.vals.insert(v, a);
                return Ok(());
            }
            if known == Some(Known::Meet) {
                let at = self.arg_lval(out, &args[0])?;
                let z = self.kf(out, 0.0);
                let st = self.cmp(out, BinOp::Gt, at.lo, z);
                let sf = self.cmp(out, BinOp::Le, at.hi, z);
                self.sure.insert(v, (st, sf));
                self.meeting.insert(v);
                return Ok(());
            }
            if known == Some(Known::Smin)
                && mask[2] == 0
                && let (Arg::Value(a), Arg::Value(b), Arg::Value(k)) =
                    (&args[0], &args[1], &args[2])
            {
                let r = self.smin(out, *a, *b, *k);
                self.vals.insert(v, r);
                return Ok(());
            }
        }
        let params = self.m.functions[g.index()].params.clone();
        let dg = derive(self.m, self.cache, g, mask.clone(), self.n, self.target)?;
        let returns = returns_active(self.m, self.cache, g, mask.clone(), Mode::Interval) != 0;
        let mut new_args = Vec::new();
        let mut extra = Vec::new();
        for ((a, bits), p) in args.iter().zip(&mask).zip(&params) {
            if *bits == 0 {
                new_args.push(a.clone());
                continue;
            }
            match a {
                Arg::Place(pl) if p.by_ref => {
                    let ty = place_type(self.m, self.f, pl)?;
                    if !self.active_root(&pl.root) || !floaty(&self.m.types, ty) {
                        return Err(Error::not_derivable("active non-float place in Lipschitz"));
                    }
                    new_args.push(a.clone());
                    let (parts, lip_p) = self.place_parts(pl)?;
                    extra.extend(parts.into_iter().map(Arg::Place));
                    if !self.is_bool_root(&pl.root) {
                        extra.push(Arg::Place(lip_p));
                    }
                }
                _ => {
                    let f = self.arg_lval(out, a)?;
                    new_args.push(self.pass(out, f.lo, p.by_ref));
                    extra.push(self.pass(out, f.hi, p.by_ref));
                    for &x in &f.j_lo {
                        extra.push(self.pass(out, x, p.by_ref));
                    }
                    for &x in &f.j_hi {
                        extra.push(self.pass(out, x, p.by_ref));
                    }
                    extra.push(self.pass(out, f.lip, p.by_ref));
                }
            }
        }
        new_args.extend(extra);
        match (v, returns) {
            (Some(v), true) => {
                let ty = self.f.value_ty(v);
                let rt = self.m.functions[dg.index()]
                    .ret
                    .ok_or_else(|| Error::internal("no lipschitz return type"))?;
                let r = self.emit(out, rt, Expr::Call(dg, new_args));
                let lo = self.emit(out, ty, Expr::Extract(r, 0));
                let hi = self.emit(out, ty, Expr::Extract(r, 1));
                let j_lo = (0..self.n)
                    .map(|i| self.emit(out, ty, Expr::Extract(r, 2 + i as u32)))
                    .collect();
                let j_hi = (0..self.n)
                    .map(|i| self.emit(out, ty, Expr::Extract(r, 2 + self.n as u32 + i as u32)))
                    .collect();
                let lip = self.emit(out, self.f32_ty, Expr::Extract(r, 2 + 2 * self.n as u32));
                self.vals.insert(v, LVal { lo, hi, j_lo, j_hi, lip });
            }
            (v, _) => out.push(let_or_eval(v, Expr::Call(dg, new_args))),
        }
        Ok(())
    }

    /// `smin(a, b, k)` (`k` inactive): convex combination of `∇a` and `∇b`, so
    /// `||∇smin(a, b, k)||₂ <= max(a.lip, b.lip)` identically everywhere!
    fn smin(&mut self, out: &mut Block, a: ValueId, b: ValueId, k: ValueId) -> LVal {
        let f = self.f32_ty;
        let aa = self.of(out, a);
        let bb = self.of(out, b);
        // Value interval: `min(a, b) - max(k - |a - b|, 0)² / (4 k)`
        let mlo = self.builtin(out, Builtin::Min, vec![aa.lo, bb.lo], f);
        let mhi = self.builtin(out, Builtin::Min, vec![aa.hi, bb.hi], f);
        let (dlo, dhi) = self.iv_sub(out, (aa.lo, aa.hi), (bb.lo, bb.hi), f);
        let (alo, ahi) = self.iv_abs(out, (dlo, dhi), f);
        let z = self.kf(out, 0.0);
        let km_hi = self.bin(out, BinOp::Sub, k, ahi, f);
        let km_lo = self.bin(out, BinOp::Sub, k, alo, f);
        let h_lo = self.builtin(out, Builtin::Max, vec![km_hi, z], f);
        let h_hi = self.builtin(out, Builtin::Max, vec![km_lo, z], f);
        let least = self.kf(out, 1.1754944e-38);
        let kk = self.builtin(out, Builtin::Max, vec![k, least], f);
        let four = self.kf(out, 4.0);
        let fk = self.bin(out, BinOp::Mul, four, kk, f);
        let h_lo2 = self.bin(out, BinOp::Mul, h_lo, h_lo, f);
        let h_hi2 = self.bin(out, BinOp::Mul, h_hi, h_hi, f);
        let q_lo = self.bin(out, BinOp::Div, h_lo2, fk, f);
        let q_hi = self.bin(out, BinOp::Div, h_hi2, fk, f);
        let vlo = self.bin(out, BinOp::Sub, mlo, q_hi, f);
        let vhi = self.bin(out, BinOp::Sub, mhi, q_lo, f);
        let (vlo, vhi) = self.pad_iv(out, vlo, vhi, f, 8.0);
        // Separated beyond blend zone `k` -> winner only!
        let a_pk = self.bin(out, BinOp::Add, aa.hi, k, f);
        let b_pk = self.bin(out, BinOp::Add, bb.hi, k, f);
        let win_a = self.cmp(out, BinOp::Le, a_pk, bb.lo);
        let win_b = self.cmp(out, BinOp::Le, b_pk, aa.lo);
        let mut j_lo = Vec::with_capacity(self.n);
        let mut j_hi = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let (hl, hh) = self.iv_hull(out, (aa.j_lo[i], aa.j_hi[i]), (bb.j_lo[i], bb.j_hi[i]), f);
            let jl = self.sel(out, win_b, bb.j_lo[i], hl);
            let jh = self.sel(out, win_b, bb.j_hi[i], hh);
            j_lo.push(self.sel(out, win_a, aa.j_lo[i], jl));
            j_hi.push(self.sel(out, win_a, aa.j_hi[i], jh));
        }
        let both_lip = self.builtin(out, Builtin::Max, vec![aa.lip, bb.lip], f);
        let lip = self.sel(out, win_b, bb.lip, both_lip);
        let lip = self.sel(out, win_a, aa.lip, lip);
        let lip = self.tighten(out, lip, &j_lo, &j_hi, f);
        LVal { lo: vlo, hi: vhi, j_lo, j_hi, lip }
    }

    /// Analytical value and gradient-norm bound for `value_noise(x)` using quintic fade C²
    /// Hessian bounds (`NOISE_HESS_DIAG = 11.55`, `NOISE_HESS_MIX = 14.07`, `|∂k| <= 3.75`).
    fn noise(&mut self, out: &mut Block, g: FuncId, x: LVal) -> R<LVal> {
        let vt = self.ty(x.lo);
        let f = self.f32_ty;
        let half = self.k(out, vt, 0.5);
        let hl = self.bin(out, BinOp::Mul, x.lo, half, vt);
        let hh = self.bin(out, BinOp::Mul, x.hi, half, vt);
        let mid = self.bin(out, BinOp::Add, hl, hh, vt);
        let reach = self.bin(out, BinOp::Sub, hh, hl, vt);
        let cap_v = self.k(out, vt, CAP);
        let reach = self.builtin(out, Builtin::Min, vec![reach, cap_v], vt);
        let vg = super::ad::value_and_gradient(self.m, self.cache, g, 0, self.target, false)?;
        let rt =
            self.m.functions[vg.index()].ret.ok_or_else(|| Error::internal("no gradient type"))?;
        let by_ref = self.m.functions[vg.index()].params[0].by_ref;
        let arg = self.pass(out, mid, by_ref);
        let r = self.emit(out, rt, Expr::Call(vg, vec![arg]));
        let val = self.emit(out, f, Expr::Extract(r, 0));
        let grad = self.emit(out, vt, Expr::Extract(r, 1));
        // Component-wise Hessian bound on `∇value_noise` over `[x.lo, x.hi]`.
        let r_yzx = self.emit(out, vt, Expr::Swizzle(reach, vec![1, 2, 0]));
        let r_zxy = self.emit(out, vt, Expr::Swizzle(reach, vec![2, 0, 1]));
        let other = self.bin(out, BinOp::Add, r_yzx, r_zxy, vt);
        let k_diag = self.k(out, vt, NOISE_HESS_DIAG);
        let k_mix = self.k(out, vt, NOISE_HESS_MIX);
        let k_round = self.k(out, vt, NOISE_ROUNDING);
        let t_diag = self.bin(out, BinOp::Mul, reach, k_diag, vt);
        let t_mix = self.bin(out, BinOp::Mul, other, k_mix, vt);
        let err0 = self.bin(out, BinOp::Add, t_diag, t_mix, vt);
        let err = self.bin(out, BinOp::Add, err0, k_round, vt);
        let gmax = self.k(out, vt, NOISE_GRAD_MAX + NOISE_ROUNDING);
        let gmin = self.k(out, vt, -(NOISE_GRAD_MAX + NOISE_ROUNDING));
        let glo0 = self.bin(out, BinOp::Sub, grad, err, vt);
        let ghi0 = self.bin(out, BinOp::Add, grad, err, vt);
        let glo = self.builtin(out, Builtin::Max, vec![glo0, gmin], vt);
        let ghi = self.builtin(out, Builtin::Min, vec![ghi0, gmax], vt);
        let glo = self.builtin(out, Builtin::Min, vec![glo, ghi], vt);
        // Value bound: `val ± (dot(|g_max|, reach) + NOISE_ROUNDING)` clamped to `[-1, 1]`.
        let (_, ga) = self.iv_abs(out, (glo, ghi), vt);
        let vrad0 = self.emit(out, f, Expr::Builtin(Builtin::Dot, vec![ga, reach]));
        let kr_f = self.kf(out, NOISE_ROUNDING);
        let vrad = self.bin(out, BinOp::Add, vrad0, kr_f, f);
        let one = self.kf(out, 1.0 + NOISE_ROUNDING);
        let none = self.kf(out, -(1.0 + NOISE_ROUNDING));
        let vlo0 = self.bin(out, BinOp::Sub, val, vrad, f);
        let vhi0 = self.bin(out, BinOp::Add, val, vrad, f);
        let vlo = self.builtin(out, Builtin::Max, vec![vlo0, none], f);
        let vhi = self.builtin(out, Builtin::Min, vec![vhi0, one], f);
        // Chain rule: `∇(noise(x)) = gᵀ J_x`, `lip <= ||g||₂ * x.lip`.
        let mut j_lo = Vec::with_capacity(self.n);
        let mut j_hi = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let p = self.iv_mul(out, (glo, ghi), (x.j_lo[i], x.j_hi[i]), vt);
            let (dl, dh) = self.iv_sum_comps(out, p, vt);
            j_lo.push(dl);
            j_hi.push(dh);
        }
        let g_norm = self.vec_len_hi(out, (glo, ghi), vt);
        let raw = self.bin(out, BinOp::Mul, g_norm, x.lip, f);
        let raw = self.pad_up(out, raw, 8.0);
        let lip = self.tighten(out, raw, &j_lo, &j_hi, f);
        Ok(LVal { lo: vlo, hi: vhi, j_lo, j_hi, lip })
    }
}
