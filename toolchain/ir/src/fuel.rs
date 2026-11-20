//! Fuel (AGENTS.md rule 2, language.md §18.11): no lane of a pipeline's invocations can do more
//! than a fixed amount of work, whatever the program does, so no dispatch or draw holds the GPU
//! past the line before the host could stop it. The host ends a run only after the fact, and a
//! GPU can't stop a lane that runs on (a draw of 32 px whose one pixel marched for seconds made
//! the M4 restart its GPU four times on 2026-11-20).
//!
//! An entry point whose work the count bounds within the fuel needs nothing: a module whose every
//! entry point is so is left as it is, its shaders byte for byte as without fuel. Otherwise each
//! invocation carries its fuel in a private `u32`, set at its entry point, and every loop that
//! could do much is charged for each pass at its top: the most one pass can do, not counting the
//! loops in it that are charged themselves. A pass the fuel can't pay leaves the loop, sets the
//! pipeline's flag (in a vertex shader, which can't write storage, it only leaves), and leaves
//! the fuel at nothing, so every charged loop after it leaves at its first pass too. What runs
//! then has no charged loop in it, so its work is bounded: a lane does at most its fuel plus
//! the work of its entry point with its charged loops taken out.
//!
//! A loop is charged unless the count bounds all its passes together within `fuel /
//! SMALL_SHARE`: a small loop runs in full and is charged to the pass of the loop it is in, as
//! straight-line code is, so a march's inner loop over a few things costs no counter of its own.

use crate::work::ExprWork;
use crate::*;
use std::collections::{HashMap, HashSet};

/// A loop whose passes together the count bounds within this share of the fuel isn't charged
/// pass by pass: it is charged whole to the pass it is in.
pub const SMALL_SHARE: u64 = 64;

/// The flag's bit that says a lane ran out of fuel (the rest of the flag is the pipeline's
/// number, from 1). A bounds check's flag is the number alone, so a fuel fault is the larger:
/// the flag keeps the largest.
pub const FUEL_BIT: u32 = 1 << 31;

/// What fuel added to a module: how many loops it charges, and the flag they set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fuelled {
    pub charged: usize,
    pub flag: ResourceId,
}

/// The flag of module `m` (`array<atomic<u32>>`) at `binding`: the one it has (a bounds check's,
/// `crate::bounds`), else a new one.
pub fn flag(m: &mut Module, binding: u32) -> ResourceId {
    if let Some(i) = m.resources.iter().position(|r| r.name == FLAG && r.binding == binding) {
        return ResourceId(i as u32);
    }
    let flag_ty = m.types.intern(TypeDef::Atomic(Scalar::U32));
    m.add_resource(Resource {
        name: FLAG.into(),
        binding,
        kind: ResourceKind::StorageReadWrite,
        ty: flag_ty,
    })
}

/// The flag's name in a module.
pub const FLAG: &str = "wrela_debug_flag";

/// Adds fuel to module `m` (flattened, as written), `fuel` work a lane, its faults setting the
/// flag at `binding` to `code | FUEL_BIT`. `None` when every entry point's work is bounded within
/// `fuel`: the module is left as it was.
pub fn add_fuel(m: &mut Module, fuel: u64, binding: u32, code: u32) -> Option<Fuelled> {
    let (each, _) = work::module_work(m);
    if each.iter().all(|e| e.work.is_some_and(|w| w <= fuel)) {
        return None;
    }
    let plan = plan(m, fuel);
    let vertex = vertex_reach(m);
    let u32_ty = m.types.u32();
    let bool_ty = m.types.bool();
    let tank = m.add_resource(Resource {
        name: "wrela_fuel".into(),
        binding: u32::MAX,
        kind: ResourceKind::Private,
        ty: u32_ty,
    });
    let flag = flag(m, binding);
    let mut charged = 0;
    for i in 0..m.functions.len() {
        let mut f = std::mem::take(&mut m.functions[i]);
        let mut body = std::mem::take(&mut f.body);
        // Charged from a local of its own (a register), kept with the invocation's tank at its
        // start, around its calls and as it returns: a tank passed by reference to a function
        // the shader compiler keeps is memory, read and written each pass.
        let here = f.new_local("wrela_fuel_here", u32_ty);
        let local = Place::root(PlaceRoot::Local(here));
        let mut cx = Charge {
            f: &mut f,
            plan: &plan,
            func: i,
            ord: 0,
            tank: local.clone(),
            u32_ty,
            bool_ty,
            charged: 0,
        };
        cx.block(&mut body);
        let n = cx.charged;
        charged += n;
        if n > 0 {
            let shared = Place::root(PlaceRoot::Resource(tank));
            let mut start = Vec::new();
            load_into(&mut f, &mut start, &shared, &local, u32_ty);
            sync(&mut f, &mut body, &shared, &local, u32_ty);
            if !matches!(body.last(), Some(Stmt::Return(_))) {
                load_into(&mut f, &mut body, &local, &shared, u32_ty);
            }
            start.append(&mut body);
            body = start;
        } else {
            f.locals.pop();
        }
        f.body = body;
        m.functions[i] = f;
    }
    // Each entry point fills its tank, and (but a vertex shader's) sets the flag as it returns
    // where its tank is empty.
    let entries: Vec<FuncId> = m.entry_points.iter().map(|e| e.function).collect();
    for &e in &entries {
        if vertex.contains(&e) {
            continue;
        }
        let f = &mut m.functions[e.index()];
        let mut body = std::mem::take(&mut f.body);
        flag_on_return(f, &mut body, tank, flag, code | FUEL_BIT, u32_ty, bool_ty);
        let ends = matches!(body.last(), Some(Stmt::Return(_)));
        if !ends {
            let mut tail = Vec::new();
            raise(f, &mut tail, tank, flag, code | FUEL_BIT, u32_ty, bool_ty);
            body.append(&mut tail);
        }
        f.body = body;
    }
    for e in entries {
        let f = &mut m.functions[e.index()];
        let mut fill = Vec::new();
        let n = f.let_(
            &mut fill,
            u32_ty,
            Expr::Const(Const::U32(fuel.min(u64::from(u32::MAX)) as u32)),
        );
        fill.push(Stmt::Store(Place::root(PlaceRoot::Resource(tank)), n));
        fill.append(&mut f.body);
        f.body = fill;
    }
    Some(Fuelled { charged, flag })
}

/// Each loop of each function, by its function and its place in a walk of the function's body
/// (a loop before the loops in it): `Some(charge)` for a loop charged each pass, `None` for a
/// small one.
fn plan(m: &Module, fuel: u64) -> HashMap<(usize, usize), Option<u64>> {
    let totals = work::loop_totals(m);
    let small = fuel / SMALL_SHARE;
    // Which loops are small: by their bodies' addresses, then by place.
    let mut is_small: HashMap<(usize, usize), u64> = HashMap::new();
    for (i, f) in m.functions.iter().enumerate() {
        let mut ord = 0;
        each_loop(&f.body, &mut ord, &mut |ord, body, _| {
            if let Some(Some(t)) = totals.get(&(body.as_ptr() as usize))
                && *t <= small
            {
                is_small.insert((i, ord), *t);
            }
        });
    }
    let mut costs = Costs { m, w: ExprWork::new(m), is_small: &is_small, calls: HashMap::new() };
    let mut plan = HashMap::new();
    for (i, f) in m.functions.iter().enumerate() {
        let mut loops = 0;
        each_loop(&f.body, &mut loops, &mut |_, _, _| {});
        for ord in 0..loops {
            let charge = if is_small.contains_key(&(i, ord)) {
                None
            } else {
                Some(costs.pass(i, f, ord).clamp(1, fuel.max(1)))
            };
            plan.insert((i, ord), charge);
        }
    }
    plan
}

/// Calls `visit(place, body, continuing)` for each loop in `b`, a loop before the loops in it,
/// counting places from `ord`.
fn each_loop<'a>(
    b: &'a [Stmt],
    ord: &mut usize,
    visit: &mut impl FnMut(usize, &'a [Stmt], &'a [Stmt]),
) {
    for s in b {
        match s {
            Stmt::If { then, else_, .. } => {
                each_loop(then, ord, visit);
                each_loop(else_, ord, visit);
            }
            Stmt::Loop { body, continuing } => {
                let here = *ord;
                *ord += 1;
                visit(here, body, continuing);
                each_loop(body, ord, visit);
                each_loop(continuing, ord, visit);
            }
            _ => {}
        }
    }
}

/// The most one pass of a loop can do, not counting the charged loops in it.
struct Costs<'a> {
    m: &'a Module,
    w: ExprWork<'a>,
    is_small: &'a HashMap<(usize, usize), u64>,
    /// Each function's most work a call, not counting its charged loops.
    calls: HashMap<usize, u64>,
}

impl Costs<'_> {
    /// One pass of loop `ord` of function `func` (`f`): its body and its continuing.
    fn pass(&mut self, func: usize, f: &Function, ord: usize) -> u64 {
        let mut found = None;
        let mut n = 0;
        each_loop(&f.body, &mut n, &mut |o, body, continuing| {
            if o == ord {
                found = Some((body, continuing));
            }
        });
        let Some((body, continuing)) = found else { return u64::MAX };
        // The places of the loops inside: they follow this one's in the walk.
        let mut inner = ord + 1;
        let a = self.block(func, f, body, &mut inner);
        let b = self.block(func, f, continuing, &mut inner);
        a.saturating_add(b)
    }

    /// The most a way through `b` can do, its small loops whole and its charged ones not at
    /// all; `ord` the place of the next loop in it.
    fn block(&mut self, func: usize, f: &Function, b: &[Stmt], ord: &mut usize) -> u64 {
        let mut sum = 0u64;
        for s in b {
            let c = match s {
                Stmt::Let(_, e) | Stmt::Eval(e) => self.expr(f, e),
                Stmt::Store(p, _) => self.store(p),
                Stmt::If { then, else_, .. } => {
                    let t = self.block(func, f, then, ord);
                    let e = self.block(func, f, else_, ord);
                    t.max(e)
                }
                Stmt::Loop { body, continuing } => {
                    let here = *ord;
                    *ord += 1;
                    // The loops inside are counted in their own places either way.
                    let mut skip = *ord;
                    let count = |b: &[Stmt]| {
                        let mut n = 0;
                        each_loop(b, &mut n, &mut |_, _, _| {});
                        n
                    };
                    skip += count(body) + count(continuing);
                    *ord = skip;
                    self.is_small.get(&(func, here)).copied().unwrap_or(0)
                }
                _ => 0,
            };
            sum = sum.saturating_add(c);
        }
        sum
    }

    fn expr(&mut self, f: &Function, e: &Expr) -> u64 {
        match e {
            Expr::Call(callee, _) => self.call(callee.index()),
            _ => self.w.of(f, e).unwrap_or(u64::MAX),
        }
    }

    /// A call of function `i`: its most work, its charged loops not counted.
    fn call(&mut self, i: usize) -> u64 {
        if let Some(&c) = self.calls.get(&i) {
            return c;
        }
        // (No recursion on the GPU: a function's calls end.)
        self.calls.insert(i, u64::MAX);
        let f = &self.m.functions[i];
        let mut ord = 0;
        let c = self.block(i, f, &f.body, &mut ord);
        self.calls.insert(i, c);
        c
    }

    /// A store's own cost (the value's is its `Let`'s): a write of storage or workgroup memory.
    fn store(&self, p: &Place) -> u64 {
        match &p.root {
            PlaceRoot::Resource(r) => match self.m.resources[r.index()].kind {
                ResourceKind::StorageReadWrite => work::MEMORY,
                ResourceKind::Workgroup => work::SHARED,
                _ => 0,
            },
            PlaceRoot::Ptr(_) | PlaceRoot::Param(_) => work::MEMORY,
            _ => 0,
        }
    }
}

/// The functions a vertex entry point reaches (they can't write the flag).
fn vertex_reach(m: &Module) -> HashSet<FuncId> {
    let mut seen = HashSet::new();
    let mut todo: Vec<FuncId> = m
        .entry_points
        .iter()
        .filter(|e| matches!(e.stage, Stage::Vertex { .. }))
        .map(|e| e.function)
        .collect();
    while let Some(f) = todo.pop() {
        if seen.insert(f) {
            todo.extend(visit::calls(&m.functions[f.index()].body));
        }
    }
    seen
}

/// `to = from`, both `u32` places, at the end of `out`.
fn load_into(f: &mut Function, out: &mut Block, from: &Place, to: &Place, u: TypeId) {
    let v = f.let_(out, u, Expr::Load(from.clone()));
    out.push(Stmt::Store(to.clone(), v));
}

/// Keeps a function's own fuel (`local`) with the invocation's tank (`shared`): written back
/// before each call and `return`, read again after each call.
fn sync(f: &mut Function, b: &mut Block, shared: &Place, local: &Place, u: TypeId) {
    let old = std::mem::take(b);
    for mut s in old {
        match &mut s {
            Stmt::If { then, else_, .. } => {
                sync(f, then, shared, local, u);
                sync(f, else_, shared, local, u);
            }
            Stmt::Loop { body, continuing } => {
                sync(f, body, shared, local, u);
                sync(f, continuing, shared, local, u);
            }
            Stmt::Return(_) => load_into(f, b, local, shared, u),
            Stmt::Let(_, Expr::Call(..)) | Stmt::Eval(Expr::Call(..)) => {
                load_into(f, b, local, shared, u);
                b.push(s);
                load_into(f, b, shared, local, u);
                continue;
            }
            _ => {}
        }
        b.push(s);
    }
}

/// Before each `return` in `b` (an entry point's), [`raise`].
fn flag_on_return(
    f: &mut Function,
    b: &mut Block,
    tank: ResourceId,
    flag: ResourceId,
    code: u32,
    u: TypeId,
    bt: TypeId,
) {
    let old = std::mem::take(b);
    for mut s in old {
        match &mut s {
            Stmt::If { then, else_, .. } => {
                flag_on_return(f, then, tank, flag, code, u, bt);
                flag_on_return(f, else_, tank, flag, code, u, bt);
            }
            Stmt::Loop { body, .. } => flag_on_return(f, body, tank, flag, code, u, bt),
            Stmt::Return(_) => raise(f, b, tank, flag, code, u, bt),
            _ => {}
        }
        b.push(s);
    }
}

/// `if fuel == 0 { atomicMax(flag[0], code) }`: the lane ran out.
fn raise(
    f: &mut Function,
    out: &mut Block,
    tank: ResourceId,
    flag: ResourceId,
    code: u32,
    u: TypeId,
    bt: TypeId,
) {
    let left = f.let_(out, u, Expr::Load(Place::root(PlaceRoot::Resource(tank))));
    let zero = f.let_(out, u, Expr::Const(Const::U32(0)));
    let empty = f.let_(out, bt, Expr::Binary(BinOp::Eq, left, zero));
    let mut then = Vec::new();
    let z = f.let_(&mut then, u, Expr::Const(Const::U32(0)));
    let c = f.let_(&mut then, u, Expr::Const(Const::U32(code)));
    let at = Place { root: PlaceRoot::Resource(flag), path: vec![Proj::Index(z)] };
    then.push(Stmt::Eval(Expr::Atomic(AtomicOp::Max, at, vec![c])));
    out.push(Stmt::If { cond: empty, then, else_: Vec::new() });
}

/// Writes the charges into one function's body.
struct Charge<'a> {
    f: &'a mut Function,
    plan: &'a HashMap<(usize, usize), Option<u64>>,
    func: usize,
    ord: usize,
    /// Where this function keeps its fuel.
    tank: Place,
    u32_ty: TypeId,
    bool_ty: TypeId,
    charged: usize,
}

impl Charge<'_> {
    fn block(&mut self, b: &mut Block) {
        for s in b.iter_mut() {
            match s {
                Stmt::If { then, else_, .. } => {
                    self.block(then);
                    self.block(else_);
                }
                Stmt::Loop { body, continuing } => {
                    let here = self.ord;
                    self.ord += 1;
                    let charge = self.plan.get(&(self.func, here)).copied().unwrap_or(Some(1));
                    self.block(body);
                    self.block(continuing);
                    if let Some(c) = charge {
                        let mut head = self.head(c);
                        head.append(body);
                        *body = head;
                        self.charged += 1;
                    }
                }
                _ => {}
            }
        }
    }

    /// A pass's charge, at its top: `if fuel <= c { fuel = 0; break } fuel -= c`. A lane that
    /// pays every pass keeps at least one, so an empty tank says it ran out, and its entry point
    /// sets the flag as it returns ([`flag_on_return`]): nothing but the charge is in the loop.
    fn head(&mut self, c: u64) -> Block {
        let (u, b) = (self.u32_ty, self.bool_ty);
        let tank = self.tank.clone();
        let mut out = Vec::new();
        let left = self.f.let_(&mut out, u, Expr::Load(tank.clone()));
        let cost =
            self.f.let_(&mut out, u, Expr::Const(Const::U32(c.min(u64::from(u32::MAX)) as u32)));
        let short = self.f.let_(&mut out, b, Expr::Binary(BinOp::Le, left, cost));
        let mut then = Vec::new();
        let zero = self.f.let_(&mut then, u, Expr::Const(Const::U32(0)));
        then.push(Stmt::Store(tank.clone(), zero));
        then.push(Stmt::Break);
        out.push(Stmt::If { cond: short, then, else_: Vec::new() });
        let rest = self.f.let_(&mut out, u, Expr::Binary(BinOp::Sub, left, cost));
        out.push(Stmt::Store(tank, rest));
        out
    }
}
