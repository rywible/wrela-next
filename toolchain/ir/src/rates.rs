//! Work every invocation does that the compiler couldn't move to an earlier rate (language.md
//! §18.13, W0014): what a flattened GPU module's loops cost each invocation, said at the loop.
//!
//! Flattening moves what it can: a loop of constants runs at the build (`const_fold`), a table of
//! constants is kept once (`const_fold::keep_tables`). What's left and costs is reported, so an
//! author can move it (docs/rates.md):
//!
//! - **A loop of constants the build's fuel ran out for** ([`Module::unfolded`]): its value is
//!   the same in every invocation and every build, but every invocation computes it.
//! - **A loop whose count is the same for the whole command**: its exit is decided by what the
//!   command gives every invocation alike (a count in the uniform block, `lights.count`), so
//!   each invocation tests it and carries the loop's code, though a pipeline for each count would
//!   have none to test.
//!
//! - **A heavy call**: a function every invocation calls doing [`HEAVY`] work or more there
//!   (eight value noises a pixel in a rock's look), the deepest such call, whoever owns the fix:
//!   the warning is how an author learns that obvious code is slow before it runs on a GPU.
//!
//! A loop is reported once it weighs [`COSTLY`] work an invocation or more (a loop of constants
//! always), counted as §18.11 counts work. Nothing in a function the author holds to a
//! `@budget(work: n)`, or called from one, is reported: the budget says what it may cost.

use crate::*;
use std::collections::{HashMap, HashSet};
use wrela_diag::Span;

/// The least work an invocation's loop must weigh to be reported: a loop that leaves most
/// invocations' work alone isn't.
pub const COSTLY: u64 = 256;

/// Why a loop is reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    /// Every value it computes is a constant; the build's fuel ran out running it.
    Constants,
    /// Its count is the same for the whole command (decided by a value that is).
    SameCount,
    /// A function every invocation calls ([`Costly::callee`]) does [`HEAVY`] work or more there,
    /// and none it calls does as much.
    Heavy,
}

/// A loop every invocation runs that the compiler couldn't move, and what it weighs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Costly {
    /// Where the loop's code starts.
    pub at: Span,
    pub why: Why,
    /// The passes it makes (`None`: no bound the count is sure of).
    pub passes: Option<u32>,
    /// Its work an invocation, every pass (`None`: no bound).
    pub work: Option<u64>,
    /// The stage it runs in: "pixel", "vertex" or "invocation".
    pub each: &'static str,
    /// For [`Why::Heavy`]: the function called, by its source name, how many times an
    /// invocation calls it here, and the whole invocation's work.
    pub callee: Option<String>,
    pub times: u64,
    pub of: Option<u64>,
}

/// The least work a function's calls at one place must do in every invocation, with the
/// invocation's whole work at least as much, to be reported (W0014, [`Why::Heavy`]): eight
/// value noises a pixel, a rock's look as first written, are about 1,900.
pub const HEAVY: u64 = 1024;

/// What an entry point's calls weigh before flattening, for [`Why::Heavy`]: found while each
/// function is its own ([`heavy_calls`]), kept by [`costly_loops`] where the flattened entry
/// point still weighs [`HEAVY`] (work done at the build doesn't count).
#[derive(Clone, Debug)]
pub struct HeavyCall {
    pub entry: usize,
    pub costly: Costly,
}

/// The deepest calls in each of `m`'s entry points (not yet flattened) that do [`HEAVY`] work
/// or more an invocation: a call that heavy whose callee calls nothing that heavy. None where
/// the caller or the callee is held to a `@budget(work: n)`. `work` is each entry point's work
/// as counted (`None`: unbounded): one under [`HEAVY`] has no such call, and isn't followed.
pub fn heavy_calls(m: &Module, work: &[Option<u64>]) -> Vec<HeavyCall> {
    let budgeted: HashSet<u32> = m
        .functions
        .iter()
        .enumerate()
        .filter(|(_, f)| f.budget.is_some_and(|b| b.work.is_some()))
        .map(|(i, _)| i as u32)
        .collect();
    let mut runs = crate::work::CallRuns::new(m);
    let mut out = Vec::new();
    for (i, e) in m.entry_points.iter().enumerate() {
        if matches!(e.stage, Stage::Compute { workgroup_size: [1, 1, 1] })
            || work.get(i).copied().flatten().is_some_and(|w| w < HEAVY)
        {
            continue;
        }
        let each = match e.stage {
            Stage::Fragment { .. } => "pixel",
            Stage::Vertex { .. } => "vertex",
            Stage::Compute { .. } => "invocation",
        };
        let mut seen = HashSet::new();
        let mut found = Vec::new();
        descend(m, &mut runs, &budgeted, e.function, Vec::new(), 1, each, &mut seen, &mut found);
        out.extend(found.into_iter().map(|costly| HeavyCall { entry: i, costly }));
    }
    out
}

/// [`heavy_calls`] in function `f`, given `known` of its arguments, called `times` times an
/// invocation: whether it found one.
#[allow(clippy::too_many_arguments)]
fn descend(
    m: &Module,
    runs: &mut crate::work::CallRuns,
    budgeted: &HashSet<u32>,
    f: FuncId,
    known: crate::work::Known,
    times: u64,
    each: &'static str,
    seen: &mut HashSet<(u32, usize)>,
    found: &mut Vec<Costly>,
) -> bool {
    if budgeted.contains(&f.0) {
        return true;
    }
    let (sites, _) = runs.of(f, &known);
    // Each call in `f`'s body: its callee and where it is.
    let mut at: HashMap<usize, (FuncId, Option<Span>)> = HashMap::new();
    let mut here = None;
    visit::walk(&m.functions[f.index()].body, &mut |s| match s {
        Stmt::At(span) => here = Some(*span),
        Stmt::Let(_, e @ Expr::Call(g, _)) | Stmt::Eval(e @ Expr::Call(g, _)) => {
            at.insert(e as *const Expr as usize, (*g, here));
        }
        _ => {}
    });
    let mut any = false;
    let mut sites: Vec<_> = sites.into_iter().collect();
    sites.sort_by_key(|(a, _)| *a);
    for (addr, r) in sites {
        let work = r.work.saturating_mul(times);
        if r.unbounded || work < HEAVY {
            continue;
        }
        let Some(&(g, span)) = at.get(&addr) else { continue };
        if !seen.insert((f.0, addr)) {
            any = true;
            continue;
        }
        any = true;
        let k = r.known.iter().max_by_key(|(_, n)| *n).map(|(k, _)| k.clone()).unwrap_or_default();
        let calls = r.times.saturating_mul(times);
        if descend(m, runs, budgeted, g, k, calls, each, seen, found) {
            continue;
        }
        let Some(span) = span else { continue };
        found.push(Costly {
            at: span,
            why: Why::Heavy,
            passes: None,
            work: Some(work),
            each,
            callee: Some(source_name(&m.functions[g.index()].name)),
            times: calls,
            of: None,
        });
    }
    any
}

/// A function's IR name without the `_<n>` and `_d<n>` tags lowering gives each instance and
/// derivation.
fn source_name(name: &str) -> String {
    let mut n = name;
    while let Some((head, tail)) = n.rsplit_once('_') {
        let digits = tail.strip_prefix('d').unwrap_or(tail);
        if head.is_empty() || digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
            break;
        }
        n = head;
    }
    n.to_string()
}

/// The costly loops of each of `m`'s entry points (flattened), once each, the costliest first;
/// none at a place in `quiet` (the statements of functions held to a `@budget`).
pub fn costly_loops(m: &Module, quiet: &HashSet<Span>, heavy: &[HeavyCall]) -> Vec<Costly> {
    let unfolded: HashSet<Span> = m.unfolded.iter().filter_map(|u| u.at).collect();
    let mut out: Vec<Costly> = Vec::new();
    let mut seen = HashSet::new();
    // The heavy calls of an entry point that, flattened, still weighs as much.
    let mut totals: HashMap<usize, Option<u64>> = HashMap::new();
    for h in heavy {
        let Some(e) = m.entry_points.get(h.entry) else { continue };
        let total =
            *totals.entry(h.entry).or_insert_with(|| crate::work::entry_work(m, e.function).work);
        if total.is_some_and(|w| w >= HEAVY) && !quiet.contains(&h.costly.at) {
            let mut c = h.costly.clone();
            c.of = total;
            if seen.insert(c.at) {
                out.push(c);
            }
        }
    }
    for e in &m.entry_points {
        // A kernel of one invocation is how GPU code runs once.
        if matches!(e.stage, Stage::Compute { workgroup_size: [1, 1, 1] }) {
            continue;
        }
        let each = match e.stage {
            Stage::Fragment { .. } => "pixel",
            Stage::Vertex { .. } => "vertex",
            Stage::Compute { .. } => "invocation",
        };
        let f = &m.functions[e.function.index()];
        let mut loops = false;
        visit::walk(&f.body, &mut |s| loops |= matches!(s, Stmt::Loop { .. }));
        if !loops {
            continue;
        }
        let mut defs: HashMap<ValueId, &Expr> = HashMap::new();
        let mut stores: HashMap<LocalId, Vec<ValueId>> = HashMap::new();
        visit::walk(&f.body, &mut |s| match s {
            Stmt::Let(v, e) => {
                defs.insert(*v, e);
            }
            Stmt::Store(p, v) if p.path.is_empty() => {
                if let Some(l) = p.root_local() {
                    stores.entry(l).or_default().push(*v);
                }
            }
            _ => {}
        });
        let mut w = Walk {
            m,
            f,
            id: e.function,
            passes: None,
            defs: &defs,
            stores: &stores,
            work: crate::work::ExprWork::new(m),
            unfolded: &unfolded,
            quiet,
            each,
            out: Vec::new(),
            in_loop: false,
        };
        w.scan(&f.body, &mut None);
        for c in w.out {
            if seen.insert(c.at) {
                out.push(c);
            }
        }
    }
    out.sort_by(|a, b| {
        let w = |c: &Costly| c.work.unwrap_or(u64::MAX);
        w(b).cmp(&w(a)).then(a.at.start.cmp(&b.at.start))
    });
    out
}

struct Walk<'a> {
    m: &'a Module,
    f: &'a Function,
    id: FuncId,
    /// The most passes each loop makes (`work::loop_passes`), counted when first wanted.
    passes: Option<HashMap<usize, u32>>,
    defs: &'a HashMap<ValueId, &'a Expr>,
    /// What's stored to each local, whole, anywhere (`a || b` keeps its value in one).
    stores: &'a HashMap<LocalId, Vec<ValueId>>,
    work: crate::work::ExprWork<'a>,
    unfolded: &'a HashSet<Span>,
    quiet: &'a HashSet<Span>,
    each: &'static str,
    out: Vec<Costly>,
    /// Inside a loop: its statements' work is counted.
    in_loop: bool,
}

impl Walk<'_> {
    /// Reports the loops of `b` that are reported; `here` is the last place seen.
    fn scan(&mut self, b: &[Stmt], here: &mut Option<Span>) {
        for s in b {
            match s {
                Stmt::At(span) => *here = Some(*span),
                Stmt::If { then, else_, .. } => {
                    self.scan(then, &mut here.clone());
                    self.scan(else_, &mut here.clone());
                }
                Stmt::Loop { body, continuing } => {
                    let at = body
                        .iter()
                        .find_map(|s| match s {
                            Stmt::At(span) => Some(*span),
                            _ => None,
                        })
                        .or(*here);
                    let quiet = at.is_none_or(|a| self.quiet.contains(&a));
                    // A loop of constants is reported whatever it weighs (its count may have no
                    // bound the work count follows); one to the command's count only with a
                    // bound, as a `@budget` can hold it to one. Either says it for what's in it.
                    if let Some(at) = at.filter(|a| self.unfolded.contains(a)) {
                        if !quiet {
                            self.say(at, Why::Constants, None, None);
                        }
                        continue;
                    }
                    if self.same_count(body) {
                        let passes = self.passes_of(body);
                        let work = self.a_loop(body, continuing, here);
                        if let (Some(at), false, true) =
                            (at, quiet, work.is_some_and(|w| w >= COSTLY))
                        {
                            self.say(at, Why::SameCount, passes, work);
                            continue;
                        }
                    }
                    self.scan(body, &mut here.clone());
                    self.scan(continuing, &mut here.clone());
                }
                _ => {}
            }
        }
    }

    fn say(&mut self, at: Span, why: Why, passes: Option<u32>, work: Option<u64>) {
        self.out.push(Costly {
            at,
            why,
            passes,
            work,
            each: self.each,
            callee: None,
            times: 1,
            of: None,
        });
    }

    /// The most passes the loop of `body` makes.
    fn passes_of(&mut self, body: &Block) -> Option<u32> {
        let (m, id) = (self.m, self.id);
        let passes = self.passes.get_or_insert_with(|| crate::work::loop_passes(m, id));
        passes.get(&(body.as_ptr() as usize)).copied()
    }

    /// The most work `b` can do (`None`: no bound), reporting its costly loops; `here` is the
    /// last place seen.
    fn block(&mut self, b: &[Stmt], here: &mut Option<Span>) -> Option<u64> {
        let mut total = Some(0u64);
        for s in b {
            let w = match s {
                Stmt::At(span) => {
                    *here = Some(*span);
                    Some(0)
                }
                // Only a loop's work is wanted: outside every loop, none is counted.
                Stmt::Let(_, e) | Stmt::Eval(e) if self.in_loop => self.work.of(self.f, e),
                Stmt::If { then, else_, .. } => {
                    let t = self.block(then, &mut here.clone());
                    let e = self.block(else_, &mut here.clone());
                    t.zip(e).map(|(t, e)| t.max(e))
                }
                Stmt::Loop { body, continuing } => self.a_loop(body, continuing, here),
                _ => Some(0),
            };
            total = total.zip(w).map(|(a, b)| a.saturating_add(b));
        }
        total
    }

    fn a_loop(&mut self, body: &Block, continuing: &Block, here: &mut Option<Span>) -> Option<u64> {
        let outer = std::mem::replace(&mut self.in_loop, true);
        let w = self.loop_work(body, continuing, here);
        self.in_loop = outer;
        w
    }

    fn loop_work(
        &mut self,
        body: &Block,
        continuing: &Block,
        here: &mut Option<Span>,
    ) -> Option<u64> {
        let passes = self.passes_of(body);
        let pass = self
            .block(body, &mut here.clone())
            .zip(self.block(continuing, &mut here.clone()))
            .map(|(a, b)| a.saturating_add(b));
        passes.zip(pass).map(|(n, w)| w.saturating_mul(u64::from(n)))
    }

    /// Whether the loop leaves on a condition the command decides: a `break` (or a `return`)
    /// under an `if` whose condition (or a part or-ed into it) compares a value the command
    /// gives every invocation alike (read from its uniform block or a read-only buffer, and not
    /// a constant) with anything.
    fn same_count(&self, body: &Block) -> bool {
        let mut memo = HashMap::new();
        let mut decides = |c: ValueId| -> bool {
            let mut stack = vec![c];
            let mut seen = HashSet::new();
            while let Some(v) = stack.pop() {
                if !seen.insert(v) {
                    continue;
                }
                match self.defs.get(&v) {
                    Some(Expr::Binary(BinOp::Or, a, b)) => stack.extend([*a, *b]),
                    Some(Expr::Select { cond, if_true, if_false }) => {
                        stack.extend([*cond, *if_true, *if_false])
                    }
                    Some(Expr::Load(p)) if p.path.is_empty() => {
                        if let Some(l) = p.root_local() {
                            stack.extend(self.stores.get(&l).into_iter().flatten().copied());
                        }
                    }
                    Some(Expr::Binary(
                        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne,
                        a,
                        b,
                    )) => {
                        if self.command(*a, &mut memo) == Some(true)
                            || self.command(*b, &mut memo) == Some(true)
                        {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
            false
        };
        let leaves = |b: &Block| b.iter().any(|s| matches!(s, Stmt::Break | Stmt::Return(_)));
        body.iter().any(|s| match s {
            Stmt::If { cond, then, else_ } => (leaves(then) || leaves(else_)) && decides(*cond),
            _ => false,
        })
    }

    /// Whether `v` is the same for every invocation, computed only from constants and what the
    /// command gives (its uniform block, a read-only buffer at such an index): `Some(true)` if
    /// it reads what the command gives, `Some(false)` if it's a constant, `None` otherwise. A
    /// part of a value built here is the part it was built of (a lister's `lights.from(p)` holds
    /// the uniform's lights beside the pixel's own point).
    fn command(&self, v: ValueId, memo: &mut HashMap<ValueId, Option<bool>>) -> Option<bool> {
        if let Some(&k) = memo.get(&v) {
            return k;
        }
        // (A cycle can't occur: values are defined before they're used.)
        let all = |me: &Self, xs: &mut dyn Iterator<Item = ValueId>, memo: &mut HashMap<_, _>| {
            let mut read = false;
            for x in xs {
                read |= me.command(x, memo)?;
            }
            Some(read)
        };
        let k = match self.defs.get(&v) {
            Some(Expr::Const(_)) => Some(false),
            Some(Expr::Load(p)) => match p.root {
                PlaceRoot::Resource(r)
                    if matches!(
                        self.m.resources[r.index()].kind,
                        ResourceKind::Uniform { .. } | ResourceKind::StorageRead
                    ) =>
                {
                    let mut idx = Vec::new();
                    p.for_each_value(&mut |x| idx.push(x));
                    all(self, &mut idx.into_iter(), memo).map(|_| true)
                }
                _ => None,
            },
            Some(Expr::Extract(x, i)) => match self.defs.get(x) {
                Some(Expr::Construct(_, parts)) => {
                    parts.get(*i as usize).and_then(|p| self.command(*p, memo))
                }
                _ => self.command(*x, memo),
            },
            Some(
                e @ (Expr::Unary(..)
                | Expr::Binary(..)
                | Expr::Convert(..)
                | Expr::Bitcast(..)
                | Expr::Splat(..)
                | Expr::Swizzle(..)
                | Expr::Construct(..)
                | Expr::Select { .. }),
            ) => {
                let mut xs = Vec::new();
                e.for_each_value(&mut |x| xs.push(x));
                all(self, &mut xs.into_iter(), memo)
            }
            _ => None,
        };
        memo.insert(v, k);
        k
    }
}

/// The places of the statements of `m`'s functions held to a `@budget(work: n)`: what
/// [`costly_loops`] doesn't report. Taken before flattening, while each function is its own;
/// inlining copies the places.
pub fn budgeted(m: &Module) -> HashSet<Span> {
    let mut out = HashSet::new();
    for f in &m.functions {
        if f.budget.is_some_and(|b| b.work.is_some()) {
            visit::walk(&f.body, &mut |s| {
                if let Stmt::At(span) = s {
                    out.insert(*span);
                }
            });
        }
    }
    out
}
