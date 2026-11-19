//! Work the same for every invocation, done once a command (language.md §18.13).
//!
//! A shader computes some values only from what the command gives every invocation alike: its
//! uniform block and constants (`same`). Each invocation computes them again, to the same
//! result, and inside a ray's march each step again: a round cone's axis over its length, a
//! smooth union's `0.25 / k`. This pass moves that work out of the invocations, in two moves:
//!
//! - **Out of a function into its callers.** A same value a function computes from its
//!   parameters (and the uniform block) becomes a parameter of its own, and each call computes
//!   it from what it passes. Where the call is in a loop, the caller now computes it in the
//!   loop: the next move, or this one again, takes it further out.
//! - **Out of the shader onto the CPU.** A same value computed only from the uniform block and
//!   constants becomes a field of the block: the CPU computes it once for the command, from the
//!   block's values, where it records the command ([`Hoisted`]'s recipe), and the shader reads it.
//!
//! What moves is the work E0712 draws its line at: a group of same values (those sharing the
//! work they're computed from) that costs at least `least` work an invocation, counted as E0712
//! counts it (`same::same_work`). Values computed from a texture or a buffer, or by operations
//! the CPU computes differently (a float's conversion to an integer, an integer's division,
//! derivatives), stay: E0712 still names what's left.
//!
//! It runs on a flattened GPU module before its memory is laid out (`gpu_memory`), so the
//! block's types are the CPU's: a field the shader reads is the CPU block's field at the same
//! place.

use crate::*;
use std::collections::{HashMap, HashSet};

/// A value the CPU computes once a command: the uniform block's field `field`, of type `ty`, is
/// `recipe` of the block's other fields. Its loads are of the block (a field, a field's field,
/// an element, a vector's component), its other statements `Let`s of pure operations, and it
/// returns the value. With `count`, the value is one a counted loop's pass computes from its
/// counter (each of a union's parts): `recipe` takes the counter (a `u32`), and the field is an
/// array of `count.1` elements, element `k` the value at counter `k` (from `count.0`), each a
/// four-component vector (16 bytes apart, as a uniform block's arrays must be) holding it from
/// its first component.
#[derive(Clone, Debug)]
pub struct Hoisted {
    pub field: u32,
    pub ty: TypeId,
    pub recipe: Function,
    pub count: Option<(u32, u32)>,
}

/// The most passes of a loop whose same values move onto the CPU, one each.
pub const MOST_PASSES: u32 = 64;

/// Moves the same work of the flattened GPU module `m` out of its invocations, as the module
/// docs say: `uniform` is its uniform block (a struct), whose type gains a field for each
/// value hoisted onto the CPU. `least` is the work an invocation, at least, that moves.
pub fn hoist(m: &mut Module, uniform: ResourceId, least: u64) -> Vec<Hoisted> {
    let TypeDef::Struct { fields, .. } = m.types.get(m.resources[uniform.index()].ty).clone()
    else {
        return Vec::new();
    };
    // A load of the whole block would read the old type: then nothing is hoisted.
    let mut whole = false;
    for f in &m.functions {
        visit::walk(&f.body, &mut |s| {
            s.for_each_place(&mut |p| {
                whole |= p.root == PlaceRoot::Resource(uniform) && p.path.is_empty();
            })
        });
    }
    if whole {
        return Vec::new();
    }
    let first = fields.len() as u32;
    let mut out: Vec<Hoisted> = Vec::new();
    let mut touched = vec![false; m.functions.len()];
    // Each round moves work a call further out; GPU code doesn't recurse, so the call graph's
    // depth bounds the rounds.
    for _ in 0..4 * m.functions.len() + 8 {
        let Some(order) = callees_first(m) else { break };
        let (bits, patterns) = same_params(m, &order);
        let mut changed = false;
        // A function some calls give same values and others not: a copy for the calls whose
        // same arguments have work to move. Then the round starts again.
        let mut copied = false;
        for &f in &order {
            for p in &patterns[f] {
                if *p != bits[f]
                    && patterns[f].len() > 1
                    && pick(m, f, p, uniform, least).iter().any(|c| c.from_params)
                {
                    copy_for(m, f, p, &bits);
                    touched.push(true);
                    copied = true;
                    break;
                }
            }
            if copied {
                break;
            }
        }
        if copied {
            continue;
        }
        for &f in &order {
            let entry = m.entry_points.iter().find(|e| e.function.index() == f);
            // A kernel of one invocation a workgroup is how GPU code runs once.
            if entry
                .is_some_and(|e| matches!(e.stage, Stage::Compute { workgroup_size: [1, 1, 1] }))
            {
                continue;
            }
            let picked = pick(m, f, &bits[f], uniform, least);
            if picked.is_empty() {
                continue;
            }
            let (lifts, hoists): (Vec<_>, Vec<_>) = picked.into_iter().partition(|c| c.from_params);
            // Every recipe from the function as it is, before any value is read from the block.
            let recipes: Vec<Function> = hoists
                .iter()
                .map(|c| recipe(&m.functions[f], &c.cone, c.value, c.counter.map(|k| k.0)))
                .collect();
            for (c, recipe) in hoists.into_iter().zip(recipes) {
                let field = first + out.len() as u32;
                let ty = m.functions[f].value_ty(c.value);
                let at =
                    Place { root: PlaceRoot::Resource(uniform), path: vec![Proj::Field(field)] };
                match c.counter {
                    None => {
                        redefine(&mut m.functions[f], c.value, Expr::Load(at));
                        out.push(Hoisted { field, ty, recipe, count: None });
                    }
                    Some((counter, start, end)) => {
                        let (s, n) = match m.types.get(ty) {
                            TypeDef::Scalar(s) => (*s, 1),
                            TypeDef::Vector(s, n) => (*s, *n),
                            _ => unreachable!("picked: a number or a vector"),
                        };
                        let wide = m.types.vector_of(s, 4);
                        let array = m.types.intern(TypeDef::Array(wide, end));
                        let at = at.with(Proj::Index(counter));
                        read_element(&mut m.functions[f], c.value, at, wide, n);
                        out.push(Hoisted { field, ty: array, recipe, count: Some((start, end)) });
                    }
                }
                touched[f] = true;
                changed = true;
            }
            // A function no code calls (an entry point) has no callers to move work into.
            let called =
                m.functions.iter().any(|g| visit::calls(&g.body).iter().any(|c| c.index() == f));
            if entry.is_none() && called && !lifts.is_empty() {
                lift(m, f, &lifts);
                touched.iter_mut().for_each(|t| *t = true);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for (f, &t) in m.functions.iter_mut().zip(&touched) {
        if t {
            crate::opt::cse(f);
            crate::opt::dce(f);
        }
    }
    if !out.is_empty() {
        let r = &m.resources[uniform.index()];
        let TypeDef::Struct { name, mut fields } = m.types.get(r.ty).clone() else {
            unreachable!("checked above")
        };
        for h in &out {
            fields.push((format!("hoisted{}", h.field - first), h.ty));
        }
        let ty = m.types.intern(TypeDef::Struct { name, fields });
        m.resources[uniform.index()].ty = ty;
    }
    out
}

/// A same value to move, and the values it's computed from (itself included), in the order
/// the function defines them.
struct Picked {
    value: ValueId,
    cone: Vec<ValueId>,
    /// Computed from the function's parameters (moved into its callers), else from the block
    /// and constants alone (moved onto the CPU).
    from_params: bool,
    /// Computed from a counted loop's counter too: the value in the cone that reads it, its
    /// first value and the value it stops at.
    counter: Option<(ValueId, u32, u32)>,
}

/// The values of `f` that read a counted loop's counter in its body, each with the counter's
/// first value and the value it stops at: a loop counted by 1 (`crate::counted`) from a known
/// start below its limit, at most [`MOST_PASSES`] passes, its counter stored nowhere but before it
/// and in its step.
fn counters(f: &Function) -> HashMap<ValueId, (LocalId, u32, u32)> {
    let consts = crate::counted::constants(f);
    let mut stores = vec![0u32; f.locals.len()];
    visit::walk(&f.body, &mut |s| {
        if let Stmt::Store(p, _) = s
            && let PlaceRoot::Local(l) = p.root
        {
            stores[l.index()] += 1;
        }
    });
    let as_u32 = |c: &Const| match *c {
        Const::U32(x) => Some(x),
        Const::I32(x) if x >= 0 => Some(x as u32),
        _ => None,
    };
    let mut out = HashMap::new();
    let mut blocks: Vec<&Block> = vec![&f.body];
    while let Some(b) = blocks.pop() {
        for (i, s) in b.iter().enumerate() {
            for inner in s.blocks() {
                blocks.push(inner);
            }
            let Stmt::Loop { body, continuing } = s else { continue };
            let Some(c) = crate::counted::counted(&consts, &b[..i], body, continuing) else {
                continue;
            };
            let (Some(start), Some(end)) = (c.start.as_ref().and_then(as_u32), as_u32(&c.limit))
            else {
                continue;
            };
            let l = c.local;
            if as_u32(&c.step) != Some(1)
                || stores[l.index()] != 2
                || end <= start
                || end - start > MOST_PASSES
            {
                continue;
            }
            visit::walk(body, &mut |s| {
                if let Stmt::Let(v, Expr::Load(p)) = s
                    && p.root_local() == Some(l)
                    && p.path.is_empty()
                {
                    out.insert(*v, (l, start, end));
                }
            });
        }
    }
    out
}

/// `f`'s definition of `v` (a number or a vector of `n` components) becomes a read of the
/// four-component vector at `at` (a block's array element), its first `n` components.
fn read_element(f: &mut Function, v: ValueId, at: Place, wide: TypeId, n: u8) {
    let w = f.new_value(wide);
    let mut at = Some(at);
    visit::expand(&mut f.body, &mut |s, out| {
        if let Stmt::Let(x, _) = &s
            && *x == v
            && let Some(at) = at.take()
        {
            out.push(Stmt::Let(w, Expr::Load(at)));
            out.push(Stmt::Let(
                v,
                if n == 1 { Expr::Extract(w, 0) } else { Expr::Swizzle(w, (0..n).collect()) },
            ));
            return;
        }
        out.push(s);
    });
}

/// The functions of `m` callees first, or `None` if a call cycles.
fn callees_first(m: &Module) -> Option<Vec<usize>> {
    let n = m.functions.len();
    let callees: Vec<Vec<usize>> = m
        .functions
        .iter()
        .map(|f| visit::calls(&f.body).iter().map(|c| c.index()).collect())
        .collect();
    // 0 unseen, 1 on the path, 2 done.
    let mut state = vec![0u8; n];
    let mut order = Vec::new();
    fn visit(i: usize, callees: &[Vec<usize>], state: &mut [u8], order: &mut Vec<usize>) -> bool {
        match state[i] {
            1 => return false,
            2 => return true,
            _ => {}
        }
        state[i] = 1;
        for &c in &callees[i] {
            if !visit(c, callees, state, order) {
                return false;
            }
        }
        state[i] = 2;
        order.push(i);
        true
    }
    for i in 0..n {
        if !visit(i, &callees, &mut state, &mut order) {
            return None;
        }
    }
    Some(order)
}

/// Which parameters of each function are given a same value by every call (a function no code
/// calls: none). Callers before callees, each caller's same values found given its own.
fn same_params(m: &Module, callees_first: &[usize]) -> (Vec<Vec<bool>>, Vec<Vec<Vec<bool>>>) {
    let n = m.functions.len();
    let mut bits: Vec<Vec<bool>> = m.functions.iter().map(|f| vec![true; f.params.len()]).collect();
    // Each function's calls' own patterns, each once.
    let mut patterns: Vec<Vec<Vec<bool>>> = vec![Vec::new(); n];
    let mut called = vec![false; n];
    for &f in callees_first.iter().rev() {
        let func = &m.functions[f];
        if !called[f] {
            bits[f].iter_mut().for_each(|b| *b = false);
        }
        let same = same::analyse(m, func, &bits[f], &HashSet::new()).0;
        visit::walk(&func.body, &mut |s| {
            let (Stmt::Let(_, Expr::Call(g, args)) | Stmt::Eval(Expr::Call(g, args))) = s else {
                return;
            };
            let g = g.index();
            called[g] = true;
            let pattern = call_pattern(m, g, args, &same);
            for (b, ok) in bits[g].iter_mut().zip(&pattern) {
                *b &= ok;
            }
            if !patterns[g].contains(&pattern) {
                patterns[g].push(pattern);
            }
        });
    }
    for (f, b) in bits.iter_mut().enumerate() {
        if !called[f] {
            b.iter_mut().for_each(|x| *x = false);
        }
        for (k, p) in m.functions[f].params.iter().enumerate() {
            if p.by_ref {
                b[k] = false;
            }
        }
    }
    (bits, patterns)
}

/// Which of a call's arguments (to function `g`) are same values, given the caller's.
fn call_pattern(m: &Module, g: usize, args: &[Arg], same: &HashSet<ValueId>) -> Vec<bool> {
    let params = &m.functions[g].params;
    (0..params.len())
        .map(|k| {
            !params[k].by_ref && matches!(args.get(k), Some(Arg::Value(v)) if same.contains(v))
        })
        .collect()
}

/// A copy of function `f` for the calls that give it `pattern`'s same arguments, which call the
/// copy from now on: where some calls give it a same value and others don't (a round cone
/// called with the uniform's last part, and in a loop with each of the others), the copy's same
/// work can move into its callers.
fn copy_for(m: &mut Module, f: usize, pattern: &[bool], bits: &[Vec<bool>]) {
    let mut g = m.functions[f].clone();
    g.name = format!("{}_same", g.name);
    m.functions.push(g);
    let copy = FuncId(m.functions.len() as u32 - 1);
    for h in 0..m.functions.len() - 1 {
        if !visit::calls(&m.functions[h].body).iter().any(|c| c.index() == f) {
            continue;
        }
        let same = same::analyse(m, &m.functions[h], &bits[h], &HashSet::new()).0;
        let mut body = std::mem::take(&mut m.functions[h].body);
        visit::walk_mut(&mut body, &mut |s| {
            if let Stmt::Let(_, Expr::Call(g, args)) | Stmt::Eval(Expr::Call(g, args)) = s
                && g.index() == f
                && call_pattern(m, f, args, &same) == pattern
            {
                *g = copy;
            }
        });
        m.functions[h].body = body;
    }
}

/// Whether type `t` may be a value that moves: one the block or a parameter holds as the CPU
/// does (a number, a vector or matrix of them, a `bool`), and no half float.
fn movable_ty(m: &Module, t: TypeId) -> bool {
    match m.types.get(t) {
        TypeDef::Scalar(s) => !matches!(s, Scalar::F16 | Scalar::F64 | Scalar::I64 | Scalar::U64),
        TypeDef::Vector(s, _) => matches!(s, Scalar::F32 | Scalar::I32 | Scalar::U32),
        TypeDef::Matrix(..) => true,
        _ => false,
    }
}

/// Whether a value of type `t` may be computed on the way to one that moves: a movable type, or
/// a struct of them.
fn inner_ty(m: &Module, t: TypeId) -> bool {
    match m.types.get(t) {
        TypeDef::Struct { fields, .. } => fields.iter().all(|(_, f)| inner_ty(m, *f)),
        TypeDef::Scalar(_) | TypeDef::Vector(..) | TypeDef::Matrix(..) => movable_ty(m, t),
        _ => false,
    }
}

/// The same values of function `f` to move: those of groups of at least `least` work an
/// invocation (as `same::same_work` groups them) that the CPU can compute as the GPU does.
fn pick(m: &Module, f: usize, bits: &[bool], uniform: ResourceId, least: u64) -> Vec<Picked> {
    let func = &m.functions[f];
    let (same, _) = same::analyse(m, func, bits, &HashSet::new());
    // Each value's definition, in order.
    let mut defs: Vec<(ValueId, &Expr)> = Vec::new();
    let mut here = None;
    let mut def_at: HashMap<ValueId, (&Expr, Option<wrela_diag::Span>)> = HashMap::new();
    visit::walk(&func.body, &mut |s| match s {
        Stmt::At(span) => here = Some(*span),
        Stmt::Let(v, e) => {
            defs.push((*v, e));
            if same.contains(v) {
                def_at.insert(*v, (e, here));
            }
        }
        _ => {}
    });
    let ty = |v: ValueId| func.value_ty(v);
    let counted_loops = counters(func);
    // The values the CPU computes as the GPU does, from parameters the calls give same values,
    // the block, constants and counted loops' counters.
    let mut region: HashSet<ValueId> = HashSet::new();
    let mut params: HashSet<ValueId> = HashSet::new();
    // The region's values computed from a counter.
    let mut counted: HashSet<ValueId> = HashSet::new();
    for &(v, e) in &defs {
        if !same.contains(&v) || !inner_ty(m, ty(v)) {
            continue;
        }
        let mut all = true;
        e.for_each_value(&mut |x| all &= region.contains(&x));
        let ok = all
            && match e {
                Expr::Const(_) | Expr::Zero(_) => true,
                Expr::Param(i) => {
                    let ok = bits.get(*i as usize).copied().unwrap_or(false);
                    if ok {
                        params.insert(v);
                    }
                    ok
                }
                Expr::Load(p) if counted_loops.contains_key(&v) => {
                    let _ = p;
                    counted.insert(v);
                    true
                }
                // (An element's index is in the region, `all` says.)
                Expr::Load(p) => p.root == PlaceRoot::Resource(uniform),
                Expr::Binary(op, a, _) => {
                    let int = |t: TypeId| match m.types.get(t) {
                        TypeDef::Scalar(s) | TypeDef::Vector(s, _) => s.is_int(),
                        _ => false,
                    };
                    !(matches!(op, BinOp::Div | BinOp::Rem) && int(ty(*a)))
                }
                // A float to an integer traps out of range on the CPU.
                Expr::Convert(x, to) => {
                    let from_float = match m.types.get(ty(*x)) {
                        TypeDef::Scalar(s) | TypeDef::Vector(s, _) => s.is_float(),
                        _ => false,
                    };
                    !(from_float && to.is_int())
                }
                Expr::Builtin(b, _) => {
                    !matches!(b, Builtin::Dpdx | Builtin::Dpdy | Builtin::Fwidth)
                }
                Expr::Unary(..)
                | Expr::Construct(..)
                | Expr::Extract(..)
                | Expr::ExtractDyn(..)
                | Expr::Splat(..)
                | Expr::Swizzle(..)
                | Expr::Bitcast(..)
                | Expr::Select { .. } => true,
                _ => false,
            };
        if ok {
            region.insert(v);
            let mut from_counter = false;
            e.for_each_value(&mut |x| from_counter |= counted.contains(&x));
            if from_counter {
                counted.insert(v);
            }
        }
    }
    // A value a counter's pass computes moves as an array of numbers or vectors of them.
    let numeric = |t: TypeId| match m.types.get(t) {
        TypeDef::Scalar(s) | TypeDef::Vector(s, _) => {
            matches!(s, Scalar::F32 | Scalar::I32 | Scalar::U32)
        }
        _ => false,
    };
    // The values the region's own definitions read.
    let inside = |region: &HashSet<ValueId>| {
        let mut read: HashMap<ValueId, bool> = HashMap::new();
        visit::walk(&func.body, &mut |s| {
            let internal = matches!(s, Stmt::Let(v, _) if region.contains(v));
            let mut mark = |x: ValueId| {
                if region.contains(&x) {
                    let e = read.entry(x).or_insert(false);
                    *e |= !internal;
                }
            };
            match s {
                Stmt::Let(_, e) | Stmt::Eval(e) => e.for_each_value(&mut mark),
                // A branch on a same condition is one every invocation takes alike: the
                // condition is no work E0712 counts (`same::same_work` reads values, not
                // branches), so it isn't moved for its own sake.
                Stmt::If { .. } => {}
                _ => s.for_each_value(&mut mark),
            }
            s.for_each_place(&mut |p| {
                for x in &p.path {
                    if let Proj::Index(i) = x {
                        if region.contains(i) {
                            read.insert(*i, true);
                        }
                    }
                }
            });
        });
        read
    };
    // The frontier: region values something outside the region reads. One of a type that can't
    // move leaves the region, and what it's computed from may be the frontier instead.
    let frontier = loop {
        let read = inside(&region);
        let front: Vec<ValueId> = defs
            .iter()
            .map(|(v, _)| *v)
            .filter(|v| read.get(v).copied().unwrap_or(false))
            .collect();
        let bad: Vec<ValueId> = front
            .iter()
            .copied()
            .filter(|v| !movable_ty(m, ty(*v)) || (counted.contains(v) && !numeric(ty(*v))))
            .collect();
        if bad.is_empty() {
            break front;
        }
        for v in bad {
            region.remove(&v);
        }
        // (What reads a value that left reads it from outside the region: recomputed.)
        let mut again = true;
        while again {
            again = false;
            for &(v, e) in &defs {
                if region.contains(&v) {
                    let mut all = true;
                    e.for_each_value(&mut |x| all &= region.contains(&x));
                    if !all {
                        region.remove(&v);
                        again = true;
                    }
                }
            }
        }
    };
    if frontier.is_empty() {
        return Vec::new();
    }
    // Each frontier value's cone, and its work, as `same::same_work` counts it.
    let def: HashMap<ValueId, &Expr> = defs.iter().copied().collect();
    let order: HashMap<ValueId, usize> =
        defs.iter().enumerate().map(|(i, (v, _))| (*v, i)).collect();
    let fixed = same::fixed_values(func, &def_at);
    let times = same::times_computed(m, FuncId(f as u32), func);
    let mut work = crate::work::ExprWork::new(m);
    let mut cost: HashMap<ValueId, Option<u64>> = HashMap::new();
    let mut cones: Vec<(ValueId, Vec<ValueId>, Vec<ValueId>, bool, Option<(ValueId, u32, u32)>)> =
        Vec::new();
    for &v in &frontier {
        let mut stack = vec![v];
        let mut cone: HashSet<ValueId> = HashSet::new();
        let mut costed = Vec::new();
        let mut from_params = false;
        while let Some(x) = stack.pop() {
            if !cone.insert(x) {
                continue;
            }
            from_params |= params.contains(&x);
            let e = def[&x];
            if !matches!(e, Expr::Load(_) | Expr::Const(_) | Expr::Zero(_) | Expr::Param(_)) {
                let k = times.get(&x).copied().unwrap_or(1);
                let w = *cost.entry(x).or_insert_with(|| {
                    if fixed.contains(&x) {
                        return Some(0);
                    }
                    work.of(func, e).map(|w| w.saturating_mul(k))
                });
                if w != Some(0) {
                    costed.push(x);
                }
            }
            e.for_each_value(&mut |y| stack.push(y));
        }
        if costed.is_empty() {
            continue;
        }
        let mut cone: Vec<ValueId> = cone.into_iter().collect();
        cone.sort_by_key(|x| order[x]);
        // At most one loop's counter, and not with a parameter: the CPU computes an array over
        // the counter, which the shader reads where the cone first reads the counter.
        let reads: Vec<(ValueId, (LocalId, u32, u32))> =
            cone.iter().filter_map(|x| counted_loops.get(x).map(|c| (*x, *c))).collect();
        let counter = match reads.first() {
            None => None,
            Some(&(x, (l, start, end))) => {
                if from_params || reads.iter().any(|(_, (k, ..))| *k != l) {
                    continue;
                }
                Some((x, start, end))
            }
        };
        cones.push((v, cone, costed, from_params, counter));
    }
    // Union-find over the cones, by the costed values they share; a group moves if its costliest
    // cone's work is at least `least`.
    let mut parent: Vec<usize> = (0..cones.len()).collect();
    fn find(p: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while p[r] != r {
            r = p[r];
        }
        p[i] = r;
        r
    }
    let mut owner: HashMap<ValueId, usize> = HashMap::new();
    for (i, c) in cones.iter().enumerate() {
        for x in &c.2 {
            match owner.get(x) {
                Some(&j) => {
                    let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                    parent[a] = b;
                }
                None => {
                    owner.insert(*x, i);
                }
            }
        }
    }
    let sum = |xs: &[ValueId]| -> Option<u64> {
        xs.iter().try_fold(0u64, |a, x| cost[x].map(|w| a.saturating_add(w)))
    };
    let mut top: HashMap<usize, Option<u64>> = HashMap::new();
    for i in 0..cones.len() {
        let r = find(&mut parent, i);
        let w = sum(&cones[i].2);
        let t = top.entry(r).or_insert(Some(0));
        *t = match (*t, w) {
            (Some(a), Some(b)) => Some(a.max(b)),
            _ => None,
        };
    }
    let mut out = Vec::new();
    for i in 0..cones.len() {
        let r = find(&mut parent, i);
        if top[&r].is_none_or(|w| w >= least) {
            let (v, cone, _, from_params, counter) = cones[i].clone();
            out.push(Picked { value: v, cone, from_params, counter });
        }
    }
    out
}

/// A function of no parameters computing `value` of `f` from its `cone` (in `f`'s order), whose
/// leaves are loads of the block and constants.
fn recipe(f: &Function, cone: &[ValueId], value: ValueId, counter: Option<ValueId>) -> Function {
    let mut r = Function::new(format!("{}_hoisted", f.name), Vec::new(), Some(f.value_ty(value)));
    let mut map: HashMap<ValueId, ValueId> = HashMap::new();
    let defs = definitions(f);
    // The counter's local, whose every read in the cone is the recipe's parameter.
    let local = counter.and_then(|c| match &defs[&c] {
        Expr::Load(p) => p.root_local(),
        _ => None,
    });
    if let Some(c) = counter {
        r.params.push(Param::value("counter", f.value_ty(c)));
    }
    for &x in cone {
        let mut e = defs[&x].clone();
        if let Expr::Load(p) = &e
            && local.is_some()
            && p.root_local() == local
        {
            e = Expr::Param(0);
        }
        e.for_each_value_mut(&mut |y| *y = map[y]);
        let nx = r.new_value(f.value_ty(x));
        r.body.push(Stmt::Let(nx, e));
        map.insert(x, nx);
    }
    r.body.push(Stmt::Return(Some(map[&value])));
    r
}

/// Each value's defining expression in `f`.
fn definitions(f: &Function) -> HashMap<ValueId, Expr> {
    let mut defs = HashMap::new();
    visit::walk(&f.body, &mut |s| {
        if let Stmt::Let(v, e) = s {
            defs.insert(*v, e.clone());
        }
    });
    defs
}

/// `f`'s definition of `v` becomes `e`.
fn redefine(f: &mut Function, v: ValueId, e: Expr) {
    let mut e = Some(e);
    visit::walk_mut(&mut f.body, &mut |s| {
        if let Stmt::Let(x, d) = s
            && *x == v
            && let Some(e) = e.take()
        {
            *d = e;
        }
    });
}

/// Moves the `picked` values of function `f` (computed from its parameters) into its callers:
/// each a parameter of `f`'s own, which every call computes from what it passes.
fn lift(m: &mut Module, f: usize, picked: &[Picked]) {
    let func = &m.functions[f];
    let defs = definitions(func);
    // The statements every call computes, in `f`'s order: the cones together.
    let mut all: Vec<ValueId> = picked.iter().flat_map(|p| p.cone.iter().copied()).collect();
    let order: HashMap<ValueId, usize> = {
        let mut o = HashMap::new();
        let mut i = 0;
        visit::walk(&func.body, &mut |s| {
            if let Stmt::Let(v, _) = s {
                o.insert(*v, i);
                i += 1;
            }
        });
        o
    };
    all.sort_by_key(|x| order[x]);
    all.dedup();
    let code: Vec<(ValueId, TypeId, Expr)> =
        all.iter().map(|&x| (x, func.value_ty(x), defs[&x].clone())).collect();
    let moved: Vec<ValueId> = picked.iter().map(|p| p.value).collect();
    // `f`: each moved value is a parameter.
    let base = m.functions[f].params.len() as u32;
    for (k, &v) in moved.iter().enumerate() {
        let t = m.functions[f].value_ty(v);
        m.functions[f].params.push(Param::value(format!("same{}", base as usize + k), t));
        redefine(&mut m.functions[f], v, Expr::Param(base + k as u32));
    }
    // Each call: the moved values computed from its arguments, and passed.
    for h in 0..m.functions.len() {
        let mut g = std::mem::take(&mut m.functions[h]);
        let mut body = std::mem::take(&mut g.body);
        visit::expand(&mut body, &mut |s, out| {
            let call = match &s {
                Stmt::Let(_, Expr::Call(c, _)) | Stmt::Eval(Expr::Call(c, _)) => *c,
                _ => {
                    out.push(s);
                    return;
                }
            };
            if call.index() != f {
                out.push(s);
                return;
            }
            let mut s = s;
            let args = match &mut s {
                Stmt::Let(_, Expr::Call(_, a)) | Stmt::Eval(Expr::Call(_, a)) => a,
                _ => unreachable!(),
            };
            let mut map: HashMap<ValueId, ValueId> = HashMap::new();
            for (x, t, e) in &code {
                if let Expr::Param(i) = e
                    && let Some(Arg::Value(a)) = args.get(*i as usize)
                {
                    map.insert(*x, *a);
                    continue;
                }
                let mut e = e.clone();
                e.for_each_value_mut(&mut |y| *y = map[y]);
                let nx = g.new_value(*t);
                out.push(Stmt::Let(nx, e));
                map.insert(*x, nx);
            }
            for v in &moved {
                args.push(Arg::Value(map[v]));
            }
            out.push(s);
        });
        g.body = body;
        m.functions[h] = g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A module whose kernel calls `g(x, p)` with `x` its uniform's field and `p` its own
    /// place; `g` computes `(x / 3) * (x / 3)` (same) and adds `p` (varying). Its uniform.
    fn module() -> (Module, ResourceId) {
        let mut m = Module::default();
        let f32_ty = m.types.f32();
        let u32_ty = m.types.u32();
        let v3u = m.types.vector_of(Scalar::U32, 3);
        let block = m.types.intern(TypeDef::Struct {
            name: "Uniforms".into(),
            fields: vec![("x".into(), f32_ty)],
        });
        let u = m.add_resource(Resource {
            name: "u".into(),
            binding: 0,
            kind: ResourceKind::Uniform { storage: false },
            ty: block,
        });
        let mut g = Function::new(
            "g",
            vec![Param::value("x", f32_ty), Param::value("p", f32_ty)],
            Some(f32_ty),
        );
        let (x, p, three, s, t, r) = (
            g.new_value(f32_ty),
            g.new_value(f32_ty),
            g.new_value(f32_ty),
            g.new_value(f32_ty),
            g.new_value(f32_ty),
            g.new_value(f32_ty),
        );
        g.body = vec![
            Stmt::Let(x, Expr::Param(0)),
            Stmt::Let(p, Expr::Param(1)),
            Stmt::Let(three, Expr::Const(Const::F32(3.0))),
            Stmt::Let(s, Expr::Binary(BinOp::Div, x, three)),
            Stmt::Let(t, Expr::Binary(BinOp::Mul, s, s)),
            Stmt::Let(r, Expr::Binary(BinOp::Add, t, p)),
            Stmt::Return(Some(r)),
        ];
        let gid = m.add_function(g);
        let mut e = Function::new("k", Vec::new(), None);
        let (id, ix, pf, xu, out) = (
            e.new_value(v3u),
            e.new_value(u32_ty),
            e.new_value(f32_ty),
            e.new_value(f32_ty),
            e.new_value(f32_ty),
        );
        e.body = vec![
            Stmt::Let(id, Expr::EntryInput(0)),
            Stmt::Let(ix, Expr::Extract(id, 0)),
            Stmt::Let(pf, Expr::Convert(ix, Scalar::F32)),
            Stmt::Let(xu, Expr::Load(Place::root(PlaceRoot::Resource(u)).with(Proj::Field(0)))),
            Stmt::Let(out, Expr::Call(gid, vec![Arg::Value(xu), Arg::Value(pf)])),
            Stmt::Return(None),
        ];
        let eid = m.add_function(e);
        m.entry_points.push(EntryPoint {
            name: "k".into(),
            stage: Stage::Compute { workgroup_size: [64, 1, 1] },
            function: eid,
            inputs: vec![BuiltinInput::GlobalInvocationId],
        });
        (m, u)
    }

    /// The same work of a callee moves into its caller as a parameter, then onto the CPU: the
    /// block gains a field the recipe computes from its first, and nothing computes `x / 3` on
    /// the GPU any more.
    #[test]
    fn a_callee_s_same_work_moves_out_through_its_caller_onto_the_cpu() {
        let (mut m, u) = module();
        let hoisted = hoist(&mut m, u, 1);
        assert_eq!(hoisted.len(), 1, "{hoisted:?}");
        assert_eq!(hoisted[0].field, 1);
        assert!(hoisted[0].count.is_none());
        assert!(crate::verify(&m).is_ok(), "{}", crate::print::print(&m));
        // `g` takes the moved value; nothing divides on the GPU.
        assert_eq!(m.functions[0].params.len(), 3);
        let mut divides = false;
        for f in &m.functions {
            visit::walk(&f.body, &mut |s| {
                divides |= matches!(s, Stmt::Let(_, Expr::Binary(BinOp::Div, ..)));
            });
        }
        assert!(!divides, "{}", crate::print::print(&m));
        // The recipe divides the block's first field.
        let mut reads = false;
        visit::walk(&hoisted[0].recipe.body, &mut |s| {
            reads |= matches!(s, Stmt::Let(_, Expr::Load(p)) if p.path == vec![Proj::Field(0)]);
        });
        assert!(reads);
        let TypeDef::Struct { fields, .. } = m.types.get(m.resources[u.index()].ty) else {
            panic!("a struct")
        };
        assert_eq!(fields.len(), 2);
    }

    /// Below `least`, nothing moves.
    #[test]
    fn little_same_work_stays() {
        let (mut m, u) = module();
        let before = m.clone();
        assert!(hoist(&mut m, u, 1_000_000).is_empty());
        assert_eq!(m.functions, before.functions);
    }
}
