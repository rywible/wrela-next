//! Constant folding (GPU, language.md §18.13): a value computed only from constants is computed
//! once, as the shader is built, not by every invocation each time it's run.
//!
//! Flattening leaves values that a function computed from its constant arguments, and the
//! elements of constant arrays at indices unrolling made constant (`normalize(CUTS[i].xyz)`).
//! Each invocation computed them again, to the same result. Folded here, they become constants
//! in the WGSL.
//!
//! What's folded: `bool`s, `i32`s, `u32`s and `f32`s, and vectors of them, through arithmetic,
//! comparisons, conversions, selects and the math builtins; parts of known vectors, arrays and
//! structs; and what a local (or a private variable) holds, where only its own stores write it.
//! A float is computed as WGSL computes it, in `f32`, a transcendental through `f64` (within
//! WGSL's error bounds for it). Where WGSL leaves a result undefined (`sqrt` of a negative, a
//! `smoothstep` whose edges meet, an integer divided by zero) or it isn't finite, the value is
//! left to the GPU.
//!
//! A loop is run however many passes it takes, loops inside it too, under [`FUEL`]: the sea's
//! look once asked for a sum over 6,561 waves of constants at every pixel (two loops, 81 passes
//! each), which a limit of 4,096 passes a loop, and none for a loop in a loop, left to the GPU:
//! its pass took 43.8 ms at 3840x2160 where the sum as a constant took 3.06. A loop the fuel
//! runs out for is left as it is, and recorded ([`Module::unfolded`]) for a warning.

use crate::*;
use std::collections::{HashMap, HashSet};

/// A value known as the shader is built: a scalar, or a vector's, array's or struct's parts.
#[derive(Clone, Debug, PartialEq)]
enum K {
    S(Const),
    P(Vec<K>),
}

/// Folds what `f` computes from constants (see the module docs). Whether it folded anything.
pub(crate) fn fold_constants(m: &mut Module, f: &mut Function) -> bool {
    let name = f.name.clone();
    let taken = crate::opt::address_taken(f);
    let private = m
        .resources
        .iter()
        .enumerate()
        .filter(|(_, r)| matches!(r.kind, ResourceKind::Private))
        .map(|(i, _)| ResourceId(i as u32))
        .collect();
    let body = std::mem::take(&mut f.body);
    let mut fx = Folder {
        types: &mut m.types,
        private,
        known: Vals::default(),
        changed: false,
        here: None,
        out_of_fuel: Vec::new(),
        exact: m.exact,
    };
    let mut st = Places::default();
    f.body = fx.block(f, body, &mut st, &taken);
    for at in fx.out_of_fuel {
        let u = Unfolded { function: name.clone(), at };
        if !m.unfolded.contains(&u) {
            m.unfolded.push(u);
        }
    }
    fx.changed
}

/// The most statements the build runs a loop of constants for, loops inside it and every pass
/// counted (each statement run one unit). Past it, the loop is left to the GPU and recorded
/// ([`Module::unfolded`]) for W0014, which says to make it a `const`. It bounds the build's time
/// as a `const`'s default fuel does (2³⁴ WASM instructions, under a second; language.md §13.1),
/// not by the same count: this IR is run by walking it, at about 15 to 20 million statements a
/// second on the Ryzen 5 5500U, so 2²⁵ is about two seconds of a build.
pub const FUEL: u64 = 1 << 25;

/// What's known of each value, by its number: a run of a loop of constants reads and writes
/// this at every statement, so it's a vector, not a map.
#[derive(Default)]
struct Vals(Vec<Option<K>>);

impl Vals {
    fn get(&self, v: &ValueId) -> Option<&K> {
        self.0.get(v.index())?.as_ref()
    }

    fn insert(&mut self, v: ValueId, k: K) {
        if v.index() >= self.0.len() {
            self.0.resize(v.index() + 1, None);
        }
        self.0[v.index()] = Some(k);
    }

    fn remove(&mut self, v: &ValueId) {
        if let Some(slot) = self.0.get_mut(v.index()) {
            *slot = None;
        }
    }
}

/// What locals and private variables hold where it's known.
#[derive(Clone, Default, PartialEq)]
struct Places {
    locals: HashMap<LocalId, K>,
    private: HashMap<ResourceId, K>,
}

impl Places {
    fn join(&mut self, other: &Places) {
        self.locals.retain(|l, k| other.locals.get(l) == Some(k));
        self.private.retain(|r, k| other.private.get(r) == Some(k));
    }
}

struct Folder<'t> {
    types: &'t mut Types,
    /// The private variables (one per invocation), which only the invocation's own stores write.
    private: HashSet<ResourceId>,
    known: Vals,
    changed: bool,
    /// The source of the statement being folded, where known.
    here: Option<wrela_diag::Span>,
    /// The loops the fuel ran out for: where each starts.
    out_of_fuel: Vec<Option<wrela_diag::Span>>,
    /// An exact build ([`Module::exact`]): only what every GPU rounds the same is folded.
    exact: bool,
}

impl Folder<'_> {
    fn block(&mut self, f: &mut Function, b: Block, st: &mut Places, taken: &[bool]) -> Block {
        let mut out = Vec::with_capacity(b.len());
        for s in b {
            match s {
                Stmt::Let(v, e) => {
                    let k = self.expr(f, &e, st);
                    if matches!(e, Expr::Call(..)) {
                        // A call might write a private variable.
                        st.private.clear();
                    }
                    match k {
                        Some(k) => {
                            let e = self.rewrite(f, v, e, &k, &mut out);
                            self.known.insert(v, k);
                            out.push(Stmt::Let(v, e));
                        }
                        None => out.push(Stmt::Let(v, e)),
                    }
                }
                Stmt::Eval(e) => {
                    if matches!(e, Expr::Call(..)) {
                        st.private.clear();
                    }
                    out.push(Stmt::Eval(e));
                }
                Stmt::Store(p, v) => {
                    self.store(&p, v, st, taken);
                    out.push(Stmt::Store(p, v));
                }
                Stmt::If { cond, then, else_ } => {
                    let mut on_else = st.clone();
                    let then = self.block(f, then, st, taken);
                    let else_ = self.block(f, else_, &mut on_else, taken);
                    st.join(&on_else);
                    out.push(Stmt::If { cond, then, else_ });
                }
                Stmt::At(span) => {
                    self.here = Some(span);
                    out.push(Stmt::At(span));
                }
                Stmt::Loop { body, continuing } => {
                    // A loop of only what's known runs now: what it leaves in its locals.
                    let at = body.iter().find_map(|s| match s {
                        Stmt::At(span) => Some(*span),
                        _ => None,
                    });
                    if let Some(left) = self.run_loop(f, &body, &continuing, st, taken, at) {
                        self.changed = true;
                        for (l, k) in left {
                            let t = f.locals[l.index()].ty;
                            let Some(v) = self.emit(f, t, &k, &mut out) else { unreachable!() };
                            out.push(Stmt::Store(Place::local(l), v));
                            st.locals.insert(l, k);
                        }
                        continue;
                    }
                    // What the loop stores holds what the pass before left: unknown in it and
                    // after it.
                    let (mut locals, mut private, mut calls) = (Vec::new(), Vec::new(), false);
                    for b in [&body, &continuing] {
                        visit::walk(b, &mut |s| {
                            if let Stmt::Store(p, _) = s {
                                match p.root {
                                    PlaceRoot::Local(l) => locals.push(l),
                                    PlaceRoot::Resource(r) => private.push(r),
                                    _ => {}
                                }
                            }
                            calls |= s.expr().is_some_and(|e| matches!(e, Expr::Call(..)));
                        });
                    }
                    let forget = |st: &mut Places| {
                        locals.iter().for_each(|l| {
                            st.locals.remove(l);
                        });
                        private.iter().for_each(|r| {
                            st.private.remove(r);
                        });
                        if calls {
                            st.private.clear();
                        }
                    };
                    forget(st);
                    let body = self.block(f, body, st, taken);
                    forget(st);
                    let continuing = self.block(f, continuing, st, taken);
                    forget(st);
                    out.push(Stmt::Loop { body, continuing });
                }
                s => out.push(s),
            }
        }
        out
    }

    /// Runs a loop whose every value is known, from what `st` knows: the locals it stores, with
    /// what each holds after it, in a kind `emit` writes. `None` (the loop left as it is) where
    /// a value isn't known, it has an effect beyond its locals' stores, or it runs past
    /// [`FUEL`] (then recorded, at `at` or where the fold is).
    fn run_loop(
        &mut self,
        f: &Function,
        body: &[Stmt],
        continuing: &[Stmt],
        st: &Places,
        taken: &[bool],
        at: Option<wrela_diag::Span>,
    ) -> Option<Vec<(LocalId, K)>> {
        let mut pl = Places { locals: st.locals.clone(), private: HashMap::new() };
        // The locals it stores, and its own values, known while it runs, forgotten after.
        let mut stored: HashSet<LocalId> = HashSet::new();
        let mut defined: Vec<ValueId> = Vec::new();
        for b in [body, continuing] {
            visit::walk(b, &mut |s| match s {
                Stmt::Let(v, _) => defined.push(*v),
                Stmt::Store(p, _) => {
                    if let Some(l) = p.root_local() {
                        stored.insert(l);
                    }
                }
                _ => {}
            });
        }
        let mut steps = 0u64;
        let ran = {
            let mut cx = RunCx { pl: &mut pl, steps: &mut steps, taken };
            self.run_loop_body(f, body, continuing, &mut cx)
        };
        for v in &defined {
            self.known.remove(v);
        }
        if steps > FUEL {
            self.out_of_fuel.push(at.or(self.here));
        }
        ran?;
        let mut stored: Vec<LocalId> = stored.into_iter().collect();
        stored.sort_by_key(|l| l.0);
        // A local the loop could store, but didn't, unknown before it, is as it was: a store
        // the run made leaves its local known (else the run fails).
        stored
            .into_iter()
            .filter_map(|l| Some((l, pl.locals.get(&l)?.clone())))
            .map(|(l, k)| writable(self.types, f.locals[l.index()].ty, &k).then_some((l, k)))
            .collect()
    }

    /// Runs a loop's passes until it breaks: `None` where it can't be run here.
    fn run_loop_body(
        &mut self,
        f: &Function,
        body: &[Stmt],
        continuing: &[Stmt],
        cx: &mut RunCx,
    ) -> Option<()> {
        loop {
            match self.run(f, body, cx)? {
                Run::Break => return Some(()),
                Run::Normal | Run::Continue => {}
            }
            if self.run(f, continuing, cx)? != Run::Normal {
                return None;
            }
        }
    }

    /// Runs `b` for `run_loop`: how it ended.
    fn run(&mut self, f: &Function, b: &[Stmt], cx: &mut RunCx) -> Option<Run> {
        for s in b {
            *cx.steps += 1;
            if *cx.steps > FUEL {
                return None;
            }
            match s {
                Stmt::At(_) => {}
                Stmt::Let(v, e) => {
                    if matches!(e, Expr::Call(..)) {
                        return None;
                    }
                    let k = self.expr(f, e, cx.pl)?;
                    self.known.insert(*v, k);
                }
                Stmt::Store(p, v) => {
                    let l = p.root_local()?;
                    if cx.taken[l.index()] {
                        return None;
                    }
                    let value = self.known.get(v)?.clone();
                    let path = self.path(p)?;
                    if !path.is_empty() && !cx.pl.locals.contains_key(&l) {
                        return None;
                    }
                    put(&mut cx.pl.locals, l, Some(value), Some(path));
                    if !cx.pl.locals.contains_key(&l) {
                        return None;
                    }
                }
                Stmt::If { cond, then, else_ } => {
                    let c = match self.known.get(cond)? {
                        K::S(Const::Bool(c)) => *c,
                        _ => return None,
                    };
                    let r = self.run(f, if c { then } else { else_ }, cx)?;
                    if r != Run::Normal {
                        return Some(r);
                    }
                }
                // A loop inside: run to its break, which leaves it alone.
                Stmt::Loop { body, continuing } => self.run_loop_body(f, body, continuing, cx)?,
                Stmt::Break => return Some(Run::Break),
                Stmt::Continue => return Some(Run::Continue),
                // A return, an effect: not run here.
                _ => return None,
            }
        }
        Some(Run::Normal)
    }

    /// Writes `k`, of type `t`, as constants: the value that holds it.
    fn emit(&mut self, f: &mut Function, t: TypeId, k: &K, out: &mut Block) -> Option<ValueId> {
        match (self.types.get(t).clone(), k) {
            (TypeDef::Scalar(_), K::S(c)) => {
                let v = f.let_(out, t, Expr::Const(c.clone()));
                self.known.insert(v, k.clone());
                Some(v)
            }
            (TypeDef::Vector(s, _), K::P(parts)) => {
                let st = self.types.intern(TypeDef::Scalar(s));
                let mut comps = Vec::new();
                for p in parts {
                    comps.push(self.emit(f, st, p, out)?);
                }
                let v = f.let_(out, t, Expr::Construct(t, comps));
                self.known.insert(v, k.clone());
                Some(v)
            }
            _ => None,
        }
    }

    /// A store's effect on what's known of its place.
    fn store(&mut self, p: &Place, v: ValueId, st: &mut Places, taken: &[bool]) {
        let value = self.known.get(&v).cloned();
        let path = self.path(p);
        match p.root {
            PlaceRoot::Local(l) if !taken[l.index()] => put(&mut st.locals, l, value, path),
            PlaceRoot::Local(l) => {
                st.locals.remove(&l);
            }
            PlaceRoot::Resource(r) if self.private.contains(&r) => {
                put(&mut st.private, r, value, path)
            }
            _ => {}
        }
    }

    /// The parts a place's path picks, where its indices are known.
    fn path(&self, p: &Place) -> Option<Vec<usize>> {
        p.path
            .iter()
            .map(|x| match x {
                Proj::Field(i) => Some(*i as usize),
                Proj::Comp(i) => Some(*i as usize),
                Proj::Index(v) => match self.known.get(v) {
                    Some(K::S(Const::U32(i))) => Some(*i as usize),
                    Some(K::S(Const::I32(i))) if *i >= 0 => Some(*i as usize),
                    _ => None,
                },
            })
            .collect()
    }

    fn read(&self, p: &Place, st: &Places) -> Option<K> {
        let mut at = match p.root {
            PlaceRoot::Local(l) => st.locals.get(&l)?,
            PlaceRoot::Resource(r) if self.private.contains(&r) => st.private.get(&r)?,
            _ => return None,
        };
        for i in self.path(p)? {
            match at {
                K::P(parts) => at = parts.get(i)?,
                K::S(_) => return None,
            }
        }
        Some(at.clone())
    }

    /// The expression `v` is given in place of `e`, now that it's known to be `k`: a constant,
    /// or a vector of constants. Other values (arrays, structs) keep their expression.
    fn rewrite(&mut self, f: &mut Function, v: ValueId, e: Expr, k: &K, out: &mut Block) -> Expr {
        match (k, &e) {
            (_, Expr::Const(_)) => e,
            (K::S(c), _) => {
                self.changed = true;
                Expr::Const(c.clone())
            }
            (K::P(parts), _) => {
                let t = f.value_ty(v);
                let TypeDef::Vector(s, _) = *self.types.get(t) else { return e };
                // Already a vector of constants: as it is.
                if let Expr::Construct(_, xs) = &e
                    && xs.iter().all(|x| matches!(self.known.get(x), Some(K::S(_))))
                {
                    return e;
                }
                let st = self.types.intern(TypeDef::Scalar(s));
                let mut comps = Vec::with_capacity(parts.len());
                for p in parts {
                    let K::S(c) = p else { return e };
                    let x = f.let_(out, st, Expr::Const(c.clone()));
                    self.known.insert(x, p.clone());
                    comps.push(x);
                }
                self.changed = true;
                Expr::Construct(t, comps)
            }
        }
    }

    /// `e`'s value, if it's known.
    fn expr(&self, f: &Function, e: &Expr, st: &Places) -> Option<K> {
        let k = |v: &ValueId| self.known.get(v).cloned();
        match e {
            Expr::Const(c) => Some(K::S(c.clone())),
            Expr::Zero(t) => zero(self.types, *t),
            Expr::Load(p) => self.read(p, st),
            Expr::Construct(t, xs) => {
                let parts = xs.iter().map(k).collect::<Option<Vec<_>>>()?;
                // A matrix's columns, an enum: not folded.
                match self.types.get(*t) {
                    TypeDef::Vector(..) | TypeDef::Array(..) | TypeDef::Struct { .. } => {
                        Some(K::P(parts))
                    }
                    _ => None,
                }
            }
            Expr::Extract(x, i) => match k(x)? {
                K::P(parts) => parts.get(*i as usize).cloned(),
                K::S(_) => None,
            },
            Expr::ExtractDyn(x, i) => {
                let K::P(parts) = k(x)? else { return None };
                let i = match k(i)? {
                    K::S(Const::U32(i)) => i as usize,
                    K::S(Const::I32(i)) if i >= 0 => i as usize,
                    _ => return None,
                };
                parts.get(i).cloned()
            }
            Expr::Splat(x, n) => {
                let K::S(c) = k(x)? else { return None };
                Some(K::P(vec![K::S(c); *n as usize]))
            }
            Expr::Swizzle(x, comps) => {
                let K::P(parts) = k(x)? else { return None };
                let picked =
                    comps.iter().map(|&c| parts.get(c as usize).cloned()).collect::<Option<_>>()?;
                Some(K::P(picked))
            }
            Expr::Unary(op, x) => each1(k(x)?, &|c| unary(*op, c)),
            Expr::Binary(op, a, b) => {
                // A matrix's product isn't folded.
                let ta = self.types.get(f.value_ty(*a));
                let tb = self.types.get(f.value_ty(*b));
                if !matches!(ta, TypeDef::Scalar(_) | TypeDef::Vector(..))
                    || !matches!(tb, TypeDef::Scalar(_) | TypeDef::Vector(..))
                {
                    return None;
                }
                // Exact: a float division or remainder is left to the GPU (Vulkan allows its
                // division 2.5 ULP), so it rounds the same whether its operands are constants.
                if self.exact
                    && matches!(op, BinOp::Div | BinOp::Rem)
                    && float_typed(self.types, f.value_ty(*a))
                {
                    return None;
                }
                let r = each2(k(a)?, k(b)?, &|x, y| binary(*op, x, y))?;
                // Exact: a float result below the smallest normal is left to the GPU too, which
                // may flush it to 0 where the fold would keep it.
                if self.exact && has_subnormal(&r) {
                    return None;
                }
                Some(r)
            }
            Expr::Convert(x, s) => each1(k(x)?, &|c| convert(c, *s)),
            Expr::Select { cond, if_true, if_false } => {
                let (c, t, e) = (k(cond)?, k(if_true)?, k(if_false)?);
                select(c, t, e)
            }
            Expr::Builtin(b, args) => {
                // Exact: only a builtin every GPU rounds as the CPU does is folded (a sine, a
                // square root, a dot product are the GPU's to round).
                if self.exact && !rounds_alike(*b) {
                    return None;
                }
                let args = args.iter().map(k).collect::<Option<Vec<_>>>()?;
                builtin(*b, &args)
            }
            _ => None,
        }
    }
}

/// What `run_loop` runs a loop with: what's known of the locals, the statements run, and the
/// locals whose addresses are taken.
struct RunCx<'a> {
    pl: &'a mut Places,
    steps: &'a mut u64,
    taken: &'a [bool],
}

/// How a block `run_loop` runs ended.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Run {
    Normal,
    Break,
    Continue,
}

/// Whether `emit` writes a `k` of type `t`: a scalar or a vector of known constants.
fn writable(types: &Types, t: TypeId, k: &K) -> bool {
    match (types.get(t), k) {
        (TypeDef::Scalar(_), K::S(_)) => true,
        (TypeDef::Vector(_, n), K::P(parts)) => {
            parts.len() == *n as usize && parts.iter().all(|p| matches!(p, K::S(_)))
        }
        _ => false,
    }
}

/// What's known of `key`'s place once `value` is stored at `path` in it (`None`: not known).
fn put<Key: std::hash::Hash + Eq>(
    map: &mut HashMap<Key, K>,
    key: Key,
    value: Option<K>,
    path: Option<Vec<usize>>,
) {
    let (Some(value), Some(path)) = (value, path) else {
        map.remove(&key);
        return;
    };
    if path.is_empty() {
        map.insert(key, value);
        return;
    }
    let Some(mut at) = map.get_mut(&key) else { return };
    for &i in &path {
        match at {
            K::P(parts) if i < parts.len() => at = &mut parts[i],
            _ => {
                map.remove(&key);
                return;
            }
        }
    }
    *at = value;
}

/// A zero of type `t`, where it's a scalar or a vector of one folded here.
fn zero(types: &Types, t: TypeId) -> Option<K> {
    let z = |s: Scalar| match s {
        Scalar::Bool => Some(Const::Bool(false)),
        Scalar::I32 => Some(Const::I32(0)),
        Scalar::U32 => Some(Const::U32(0)),
        Scalar::F32 => Some(Const::F32(0.0)),
        _ => None,
    };
    match types.get(t) {
        TypeDef::Scalar(s) => z(*s).map(K::S),
        TypeDef::Vector(s, n) => Some(K::P(vec![K::S(z(*s)?); *n as usize])),
        _ => None,
    }
}

fn each1(x: K, op: &dyn Fn(&Const) -> Option<Const>) -> Option<K> {
    match x {
        K::S(c) => op(&c).map(K::S),
        K::P(xs) => xs.into_iter().map(|x| each1(x, op)).collect::<Option<Vec<_>>>().map(K::P),
    }
}

fn each2(a: K, b: K, op: &dyn Fn(&Const, &Const) -> Option<Const>) -> Option<K> {
    match (a, b) {
        (K::S(x), K::S(y)) => op(&x, &y).map(K::S),
        (K::P(xs), K::P(ys)) if xs.len() == ys.len() => xs
            .into_iter()
            .zip(ys)
            .map(|(x, y)| each2(x, y, op))
            .collect::<Option<Vec<_>>>()
            .map(K::P),
        (K::S(x), K::P(ys)) => {
            ys.into_iter().map(|y| each2(K::S(x.clone()), y, op)).collect::<Option<_>>().map(K::P)
        }
        (K::P(xs), K::S(y)) => {
            xs.into_iter().map(|x| each2(x, K::S(y.clone()), op)).collect::<Option<_>>().map(K::P)
        }
        _ => None,
    }
}

fn each3(a: K, b: K, c: K, op: &dyn Fn(&Const, &Const, &Const) -> Option<Const>) -> Option<K> {
    match (a, b, c) {
        (K::S(x), K::S(y), K::S(z)) => op(&x, &y, &z).map(K::S),
        (a, b, c) => {
            let n = [&a, &b, &c].iter().find_map(|k| match k {
                K::P(p) => Some(p.len()),
                K::S(_) => None,
            })?;
            let part = |k: &K, i: usize| match k {
                K::P(p) => p.get(i).cloned(),
                K::S(_) => Some(k.clone()),
            };
            (0..n)
                .map(|i| each3(part(&a, i)?, part(&b, i)?, part(&c, i)?, op))
                .collect::<Option<Vec<_>>>()
                .map(K::P)
        }
    }
}

/// A float result, where it's finite.
fn float(x: f32) -> Option<Const> {
    x.is_finite().then_some(Const::F32(x))
}

fn unary(op: UnOp, c: &Const) -> Option<Const> {
    Some(match (op, c) {
        (UnOp::Neg, Const::F32(x)) => Const::F32(-x),
        (UnOp::Neg, Const::I32(x)) => Const::I32(x.wrapping_neg()),
        (UnOp::Not, Const::Bool(b)) => Const::Bool(!b),
        (UnOp::Not, Const::I32(x)) => Const::I32(!x),
        (UnOp::Not, Const::U32(x)) => Const::U32(!x),
        _ => return None,
    })
}

pub(crate) fn binary(op: BinOp, a: &Const, b: &Const) -> Option<Const> {
    use BinOp as B;
    Some(match (a, b) {
        (Const::F32(x), Const::F32(y)) => {
            let (x, y) = (*x, *y);
            match op {
                B::Add => float(x + y)?,
                B::Sub => float(x - y)?,
                B::Mul => float(x * y)?,
                B::Div => float(x / y)?,
                // WGSL's (and the GPU's `frem`): `x - y * trunc(x / y)`, in f32, not an exact
                // `fmod`; they part where `x / y` rounds to a whole number.
                B::Rem => float(x - y * (x / y).trunc())?,
                B::Eq => Const::Bool(x == y),
                B::Ne => Const::Bool(x != y),
                B::Lt => Const::Bool(x < y),
                B::Le => Const::Bool(x <= y),
                B::Gt => Const::Bool(x > y),
                B::Ge => Const::Bool(x >= y),
                _ => return None,
            }
        }
        (Const::I32(x), Const::I32(y)) => {
            let (x, y) = (*x, *y);
            match op {
                B::Add | B::WrappingAdd => Const::I32(x.wrapping_add(y)),
                B::Sub | B::WrappingSub => Const::I32(x.wrapping_sub(y)),
                B::Mul | B::WrappingMul => Const::I32(x.wrapping_mul(y)),
                B::Div if y != 0 => Const::I32(x.wrapping_div(y)),
                B::Rem if y != 0 => Const::I32(x.wrapping_rem(y)),
                B::BitAnd => Const::I32(x & y),
                B::BitOr => Const::I32(x | y),
                B::BitXor => Const::I32(x ^ y),
                B::Eq => Const::Bool(x == y),
                B::Ne => Const::Bool(x != y),
                B::Lt => Const::Bool(x < y),
                B::Le => Const::Bool(x <= y),
                B::Gt => Const::Bool(x > y),
                B::Ge => Const::Bool(x >= y),
                _ => return None,
            }
        }
        (Const::U32(x), Const::U32(y)) => {
            let (x, y) = (*x, *y);
            match op {
                B::Add | B::WrappingAdd => Const::U32(x.wrapping_add(y)),
                B::Sub | B::WrappingSub => Const::U32(x.wrapping_sub(y)),
                B::Mul | B::WrappingMul => Const::U32(x.wrapping_mul(y)),
                B::Div if y != 0 => Const::U32(x / y),
                B::Rem if y != 0 => Const::U32(x % y),
                B::BitAnd => Const::U32(x & y),
                B::BitOr => Const::U32(x | y),
                B::BitXor => Const::U32(x ^ y),
                B::Shl if y < 32 => Const::U32(x << y),
                B::Shr if y < 32 => Const::U32(x >> y),
                B::Eq => Const::Bool(x == y),
                B::Ne => Const::Bool(x != y),
                B::Lt => Const::Bool(x < y),
                B::Le => Const::Bool(x <= y),
                B::Gt => Const::Bool(x > y),
                B::Ge => Const::Bool(x >= y),
                _ => return None,
            }
        }
        (Const::I32(x), Const::U32(y)) => match op {
            B::Shl if *y < 32 => Const::I32(x.wrapping_shl(*y)),
            B::Shr if *y < 32 => Const::I32(x >> y),
            _ => return None,
        },
        (Const::Bool(x), Const::Bool(y)) => match op {
            B::And => Const::Bool(*x && *y),
            B::Or => Const::Bool(*x || *y),
            B::Eq => Const::Bool(x == y),
            B::Ne => Const::Bool(x != y),
            _ => return None,
        },
        _ => return None,
    })
}

fn convert(c: &Const, to: Scalar) -> Option<Const> {
    Some(match (c, to) {
        (Const::F32(x), Scalar::F32) => Const::F32(*x),
        (Const::I32(x), Scalar::F32) => Const::F32(*x as f32),
        (Const::U32(x), Scalar::F32) => Const::F32(*x as f32),
        // Only a float in the integer's range: past it, backends clamp to the largest float
        // the integer holds, not the integer's own bound (Rust's `as`); NaN isn't folded.
        (Const::F32(x), Scalar::I32) if (-2147483648.0..2147483648.0).contains(x) => {
            Const::I32(*x as i32)
        }
        (Const::F32(x), Scalar::U32) if *x > -1.0 && *x < 4294967296.0 => Const::U32(*x as u32),
        (Const::I32(x), Scalar::U32) => Const::U32(*x as u32),
        (Const::U32(x), Scalar::I32) => Const::I32(*x as i32),
        (Const::I32(x), Scalar::I32) => Const::I32(*x),
        (Const::U32(x), Scalar::U32) => Const::U32(*x),
        (Const::Bool(b), Scalar::F32) => Const::F32(f32::from(u8::from(*b))),
        (Const::Bool(b), Scalar::I32) => Const::I32(i32::from(*b)),
        (Const::Bool(b), Scalar::U32) => Const::U32(u32::from(*b)),
        (Const::Bool(b), Scalar::Bool) => Const::Bool(*b),
        _ => return None,
    })
}

fn select(c: K, t: K, e: K) -> Option<K> {
    match c {
        K::S(Const::Bool(b)) => Some(if b { t } else { e }),
        // A vector condition picks each component.
        K::P(cs) => {
            let (K::P(ts), K::P(es)) = (t, e) else { return None };
            cs.into_iter()
                .zip(ts.into_iter().zip(es))
                .map(|(c, (t, e))| select(c, t, e))
                .collect::<Option<Vec<_>>>()
                .map(K::P)
        }
        K::S(_) => None,
    }
}

fn f(c: &Const) -> Option<f32> {
    match c {
        Const::F32(x) => Some(*x),
        _ => None,
    }
}

/// A float's components, a vector's or a scalar's.
fn floats(k: &K) -> Option<Vec<f32>> {
    match k {
        K::S(c) => Some(vec![f(c)?]),
        K::P(ps) => ps
            .iter()
            .map(|p| match p {
                K::S(c) => f(c),
                K::P(_) => None,
            })
            .collect(),
    }
}

/// Whether every GPU gives builtin `b` the bits the CPU's folding does: what selects, compares or
/// rounds to an integer, never what computes a new rounding of its own.
fn rounds_alike(b: Builtin) -> bool {
    use Builtin as B;
    matches!(
        b,
        B::Abs
            | B::Min
            | B::Max
            | B::Clamp
            | B::Saturate
            | B::Floor
            | B::Ceil
            | B::Round
            | B::Trunc
            | B::Sign
            | B::Step
            | B::AllEqual
            | B::CountOnes
            | B::LeadingZeros
            | B::TrailingZeros
    )
}

/// Whether a folded value holds an `f32` subnormal (nonzero, below the smallest normal).
fn has_subnormal(k: &K) -> bool {
    match k {
        K::S(Const::F32(x)) => x.is_subnormal(),
        K::S(_) => false,
        K::P(ps) => ps.iter().any(has_subnormal),
    }
}

/// Whether a value of type `t` is a float scalar or vector.
fn float_typed(types: &Types, t: TypeId) -> bool {
    matches!(
        types.get(t),
        TypeDef::Scalar(Scalar::F32 | Scalar::F16) | TypeDef::Vector(Scalar::F32 | Scalar::F16, _)
    )
}

fn builtin(b: Builtin, args: &[K]) -> Option<K> {
    use Builtin as B;
    // Through `f64`, rounded once: within WGSL's error bound for each.
    let wide = |g: fn(f64) -> f64| move |c: &Const| float(g(f64::from(f(c)?)) as f32);
    match b {
        B::Sqrt => each1(args[0].clone(), &|c| {
            let x = f(c)?;
            if x < 0.0 { None } else { float(x.sqrt()) }
        }),
        B::InverseSqrt => each1(args[0].clone(), &|c| {
            let x = f(c)?;
            if x <= 0.0 { None } else { float((1.0 / f64::from(x).sqrt()) as f32) }
        }),
        B::Sin => each1(args[0].clone(), &wide(f64::sin)),
        B::Cos => each1(args[0].clone(), &wide(f64::cos)),
        B::Tan => each1(args[0].clone(), &wide(f64::tan)),
        B::Asin | B::Acos => each1(args[0].clone(), &|c| {
            let x = f64::from(f(c)?);
            if !(-1.0..=1.0).contains(&x) {
                return None;
            }
            float(if b == B::Asin { x.asin() } else { x.acos() } as f32)
        }),
        B::Atan => each1(args[0].clone(), &wide(f64::atan)),
        B::Atan2 => each2(args[0].clone(), args[1].clone(), &|y, x| {
            float(f64::from(f(y)?).atan2(f64::from(f(x)?)) as f32)
        }),
        B::Exp => each1(args[0].clone(), &wide(f64::exp)),
        B::Exp2 => each1(args[0].clone(), &wide(f64::exp2)),
        B::Log | B::Log2 => each1(args[0].clone(), &|c| {
            let x = f64::from(f(c)?);
            if x <= 0.0 {
                return None;
            }
            float(if b == B::Log { x.ln() } else { x.log2() } as f32)
        }),
        B::Sinh => each1(args[0].clone(), &wide(f64::sinh)),
        B::Cosh => each1(args[0].clone(), &wide(f64::cosh)),
        B::Tanh => each1(args[0].clone(), &wide(f64::tanh)),
        B::Pow => each2(args[0].clone(), args[1].clone(), &|x, y| {
            let (x, y) = (f64::from(f(x)?), f64::from(f(y)?));
            // WGSL's `pow` is `exp2(y * log2(x))`: undefined for x < 0, and x = 0 with y <= 0.
            if x < 0.0 || (x == 0.0 && y <= 0.0) {
                return None;
            }
            float(x.powf(y) as f32)
        }),
        B::Floor => each1(args[0].clone(), &|c| float(f(c)?.floor())),
        B::Ceil => each1(args[0].clone(), &|c| float(f(c)?.ceil())),
        B::Round => each1(args[0].clone(), &|c| float(f(c)?.round_ties_even())),
        B::Trunc => each1(args[0].clone(), &|c| float(f(c)?.trunc())),
        B::Fract => each1(args[0].clone(), &|c| {
            let x = f(c)?;
            float(x - x.floor())
        }),
        B::Abs => each1(args[0].clone(), &|c| match c {
            Const::F32(x) => float(x.abs()),
            Const::I32(x) => Some(Const::I32(x.wrapping_abs())),
            Const::U32(x) => Some(Const::U32(*x)),
            _ => None,
        }),
        B::Sign => each1(args[0].clone(), &|c| match c {
            Const::F32(x) => float(if *x > 0.0 {
                1.0
            } else if *x < 0.0 {
                -1.0
            } else {
                0.0
            }),
            Const::I32(x) => Some(Const::I32(x.signum())),
            _ => None,
        }),
        B::Min | B::Max => each2(args[0].clone(), args[1].clone(), &|x, y| {
            let max = b == B::Max;
            match (x, y) {
                // `y` where it's more (less), else `x` (§11.4).
                (Const::F32(x), Const::F32(y)) => {
                    float(if (max && y > x) || (!max && y < x) { *y } else { *x })
                }
                (Const::I32(x), Const::I32(y)) => {
                    Some(Const::I32(if max { *x.max(y) } else { *x.min(y) }))
                }
                (Const::U32(x), Const::U32(y)) => {
                    Some(Const::U32(if max { *x.max(y) } else { *x.min(y) }))
                }
                _ => None,
            }
        }),
        B::Clamp => {
            each3(args[0].clone(), args[1].clone(), args[2].clone(), &|x, lo, hi| match (x, lo, hi)
            {
                (Const::F32(x), Const::F32(lo), Const::F32(hi)) if lo <= hi => {
                    let m = if lo > x { *lo } else { *x };
                    float(if *hi < m { *hi } else { m })
                }
                (Const::I32(x), Const::I32(lo), Const::I32(hi)) if lo <= hi => {
                    Some(Const::I32(*x.max(lo).min(hi)))
                }
                (Const::U32(x), Const::U32(lo), Const::U32(hi)) if lo <= hi => {
                    Some(Const::U32(*x.max(lo).min(hi)))
                }
                _ => None,
            })
        }
        B::Saturate => each1(args[0].clone(), &|c| float(f(c)?.clamp(0.0, 1.0))),
        B::Mix => each3(args[0].clone(), args[1].clone(), args[2].clone(), &|a, b, t| {
            let (a, b, t) = (f(a)?, f(b)?, f(t)?);
            float(a * (1.0 - t) + b * t)
        }),
        B::Step => each2(args[0].clone(), args[1].clone(), &|edge, x| {
            float(if f(x)? >= f(edge)? { 1.0 } else { 0.0 })
        }),
        B::Smoothstep => each3(args[0].clone(), args[1].clone(), args[2].clone(), &|lo, hi, x| {
            let (lo, hi, x) = (f(lo)?, f(hi)?, f(x)?);
            if lo == hi {
                return None;
            }
            let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
            float(t * t * (3.0 - 2.0 * t))
        }),
        B::Length => {
            let v = floats(&args[0])?;
            float(v.iter().map(|x| x * x).sum::<f32>().sqrt()).map(K::S)
        }
        B::Distance => {
            let (a, b) = (floats(&args[0])?, floats(&args[1])?);
            (a.len() == b.len()).then_some(())?;
            float(a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()).map(K::S)
        }
        B::Dot => {
            let (a, b) = (floats(&args[0])?, floats(&args[1])?);
            (a.len() == b.len()).then_some(())?;
            float(a.iter().zip(&b).map(|(x, y)| x * y).sum()).map(K::S)
        }
        B::Cross => {
            let (a, b) = (floats(&args[0])?, floats(&args[1])?);
            (a.len() == 3 && b.len() == 3).then_some(())?;
            let c =
                [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
            c.iter().map(|x| float(*x).map(K::S)).collect::<Option<Vec<_>>>().map(K::P)
        }
        B::Normalize => {
            let v = floats(&args[0])?;
            let l = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if l == 0.0 || v.len() < 2 {
                return None;
            }
            v.iter().map(|x| float(x / l).map(K::S)).collect::<Option<Vec<_>>>().map(K::P)
        }
        _ => None,
    }
}

// ---- constant tables, kept once ----------------------------------------------------------------

/// Keeps each constant table `f` reads at an index it works out as the module's constant data
/// (language.md §18.13), and has those reads read the data: whether it kept any.
///
/// A table built of constants and read at a worked-out index (`REST[THIGH_LEFT + 2 * seg +
/// side]`) was built in each invocation's own memory, each time it was read: the cloth's 57
/// joints, twice a node a pass, took its 18 passes 6.6 ms at 3840x2160 on the M4, 0.22 with each
/// joint its own constant. Kept once, in the shader's constant memory, a read is a read. A table
/// whose reads are all at constant indices was folded already, and one of a type constant data
/// doesn't hold here (a `bool`, an enum, a matrix) stays as it is.
pub(crate) fn keep_tables(m: &mut Module, f: &mut Function) -> bool {
    // What each value is, where it's a constant: one walk, in order (a value is defined before
    // it's used).
    let known = {
        let mut fx = Folder {
            types: &mut m.types,
            private: HashSet::new(),
            known: Vals::default(),
            changed: false,
            here: None,
            out_of_fuel: Vec::new(),
            exact: m.exact,
        };
        let none = Places::default();
        visit::walk(&f.body, &mut |s| {
            if let Stmt::Let(v, e) = s
                && !matches!(e, Expr::Load(_))
                && let Some(k) = fx.expr(f, e, &none)
            {
                fx.known.insert(*v, k);
            }
        });
        fx.known
    };
    let table = |types: &Types, t: TypeId, v: ValueId| -> Option<K> {
        let k = known.get(&v)?;
        (matches!(types.get(t), TypeDef::Array(..)) && holds(types, t, k)).then(|| k.clone())
    };
    // The locals that hold a table: every store to one stores the same table, whole, and every
    // other mention of it is a read.
    let taken = crate::opt::address_taken(f);
    let mut held: Vec<Option<K>> = vec![None; f.locals.len()];
    let mut not: Vec<bool> = taken.clone();
    let mut dynamic: Vec<bool> = vec![false; f.locals.len()];
    let is_const = |v: &ValueId| matches!(known.get(v), Some(K::S(_)));
    visit::walk(&f.body, &mut |s| match s {
        Stmt::Store(p, v) => {
            if let PlaceRoot::Local(l) = p.root {
                let t = f.locals[l.index()].ty;
                match (p.path.is_empty(), table(&m.types, t, *v)) {
                    (true, Some(k)) if held[l.index()].as_ref().is_none_or(|h| *h == k) => {
                        held[l.index()] = Some(k)
                    }
                    _ => not[l.index()] = true,
                }
            }
        }
        Stmt::Let(_, Expr::Load(p)) => {
            if let PlaceRoot::Local(l) = p.root
                && p.path.iter().any(|x| matches!(x, Proj::Index(i) if !is_const(i)))
            {
                dynamic[l.index()] = true;
            }
        }
        s => {
            if let Some(e) = s.expr() {
                e.for_each_place(&mut |p| {
                    if let Some(l) = p.root_local() {
                        not[l.index()] = true;
                    }
                });
            }
        }
    });
    // A read before the table's stored there would read the local's zeros.
    unstored_reads(&f.body, &mut Vec::new(), &mut not);
    // Each table kept once a module, however many functions and locals read it.
    let data_of = |m: &mut Module, t: TypeId, k: &K| -> DataId {
        let value = const_value(k);
        if let Some(i) = m.data.iter().position(|d| d.ty == t && d.value == value) {
            return DataId(i as u32);
        }
        let name = format!("table{}", m.data.len());
        m.add_data(Data { name, ty: t, value })
    };
    let mut kept: Vec<Option<DataId>> = vec![None; f.locals.len()];
    for l in 0..f.locals.len() {
        if let (Some(k), false, true) = (&held[l], not[l], dynamic[l]) {
            kept[l] = Some(data_of(m, f.locals[l].ty, k));
        }
    }
    // A table value read at a worked-out index.
    let mut extracts: HashMap<ValueId, DataId> = HashMap::new();
    visit::walk(&f.body, &mut |s| {
        if let Stmt::Let(_, Expr::ExtractDyn(x, i)) = s
            && !is_const(i)
            && let Some(k) = table(&m.types, f.value_ty(*x), *x)
        {
            let t = f.value_ty(*x);
            extracts.entry(*x).or_insert_with(|| data_of(m, t, &k));
        }
    });
    if kept.iter().all(Option::is_none) && extracts.is_empty() {
        return false;
    }
    visit::walk_mut(&mut f.body, &mut |s| match s {
        Stmt::Let(_, Expr::Load(p)) => {
            if let PlaceRoot::Local(l) = p.root
                && let Some(d) = kept[l.index()]
            {
                p.root = PlaceRoot::Data(d);
            }
        }
        Stmt::Let(_, e @ Expr::ExtractDyn(..)) => {
            if let Expr::ExtractDyn(x, i) = *e
                && let Some(&d) = extracts.get(&x)
            {
                *e = Expr::Load(Place { root: PlaceRoot::Data(d), path: vec![Proj::Index(i)] });
            }
        }
        _ => {}
    });
    true
}

/// Marks in `not` each local some load of which no store before it in its block, or in a block
/// around it, has stored (`stored`: the locals so stored, a stack).
fn unstored_reads(b: &[Stmt], stored: &mut Vec<LocalId>, not: &mut [bool]) {
    let depth = stored.len();
    for s in b {
        match s {
            Stmt::Store(p, _) if p.path.is_empty() => {
                if let Some(l) = p.root_local() {
                    stored.push(l);
                }
            }
            Stmt::Let(_, e) => e.for_each_place(&mut |p| {
                if let Some(l) = p.root_local()
                    && !stored.contains(&l)
                {
                    not[l.index()] = true;
                }
            }),
            s => {
                for inner in s.blocks() {
                    unstored_reads(inner, stored, not);
                }
            }
        }
    }
    stored.truncate(depth);
}

/// Whether constant data holds `k`, of type `t`, here: scalars of 32 bits that aren't `bool`s
/// (memory holds a `bool` as a `u32`: `gpu_memory`), in vectors, arrays and structs.
fn holds(types: &Types, t: TypeId, k: &K) -> bool {
    match (types.get(t), k) {
        (TypeDef::Scalar(s), K::S(_)) => matches!(s, Scalar::F32 | Scalar::I32 | Scalar::U32),
        (TypeDef::Vector(s, n), K::P(parts)) => {
            matches!(s, Scalar::F32 | Scalar::I32 | Scalar::U32)
                && parts.len() == *n as usize
                && parts.iter().all(|p| matches!(p, K::S(_)))
        }
        (TypeDef::Array(e, n), K::P(parts)) => {
            parts.len() == *n as usize && parts.iter().all(|p| holds(types, *e, p))
        }
        (TypeDef::Struct { fields, .. }, K::P(parts)) => {
            parts.len() == fields.len()
                && fields.iter().zip(parts).all(|((_, ft), p)| holds(types, *ft, p))
        }
        _ => false,
    }
}

/// `k` as the IR's constant value.
fn const_value(k: &K) -> ConstValue {
    match k {
        K::S(c) => ConstValue::Scalar(c.clone()),
        K::P(parts) => ConstValue::Parts(parts.iter().map(const_value).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether `x * y` of two constants folds, in an exact build or not.
    fn folds(x: f32, y: f32, exact: bool) -> bool {
        let mut m = Module { exact, ..Module::default() };
        let f32t = m.types.f32();
        let mut f = Function::new("product", Vec::new(), Some(f32t));
        let mut body = Vec::new();
        let a = f.let_(&mut body, f32t, Expr::Const(Const::F32(x)));
        let b = f.let_(&mut body, f32t, Expr::Const(Const::F32(y)));
        let p = f.let_(&mut body, f32t, Expr::Binary(BinOp::Mul, a, b));
        body.push(Stmt::Return(Some(p)));
        f.body = body;
        fold_constants(&mut m, &mut f);
        let mut left = false;
        visit::walk(&f.body, &mut |s| left |= matches!(s, Stmt::Let(_, Expr::Binary(..))));
        !left
    }

    /// An exact build leaves a product whose result is subnormal to the GPU, which may flush it
    /// to 0; a normal one it folds, and a build that isn't exact folds both.
    #[test]
    fn an_exact_build_leaves_a_subnormal_result_to_the_gpu() {
        assert!(folds(1.5, 2.0, true));
        assert!(!folds(1.0e-20, 1.0e-20, true));
        assert!(folds(1.0e-20, 1.0e-20, false));
    }
}
