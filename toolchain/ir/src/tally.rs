//! A counting build's tally (docs/plans/engine-at-build.md phase 4 step 5): each invocation adds
//! up the work it does, as the work count counts it (`crate::work`), and as it returns adds its
//! sum into its pipeline's two words of the debug flag's buffer (`crate::fuel::flag`), the low
//! word and the carries out of it, so the host can read how much work each pipeline did in a
//! frame: what a pass costs in a usual frame, not the bound of its every invocation.
//!
//! The sum is the work of each stretch of code that ran: a straight run of statements adds its
//! own work where it ends (before a branch, a loop, a `break`, a `continue` or a `return`, and at
//! the end of its block), a branch's test is its run's, each pass of a loop adds its body's and
//! its continuing's runs, and a call adds nothing itself: the function it calls adds its own.
//! So the tally is the count's work of the path the invocation took, exactly as the count would
//! weigh that path alone. A vertex shader can't write storage: its work isn't tallied.

use crate::work::ExprWork;
use crate::*;

/// The flag buffer's words a pipeline's tally takes: its low word and the carries out of it.
pub const TALLY_WORDS: u32 = 2;

/// The first of pipeline `code`'s words in the flag buffer (`code` from 1; word 0 is the flag).
pub fn tally_word(code: u32) -> u32 {
    1 + (code - 1) * TALLY_WORDS
}

/// Adds the tally to module `m`, whose flag is at `binding`, for pipeline `code` (from 1).
/// Whether any entry point is tallied (a module of vertex shaders alone isn't).
pub fn add_tally(m: &mut Module, binding: u32, code: u32) -> bool {
    let vertex: std::collections::HashSet<FuncId> = m
        .entry_points
        .iter()
        .filter(|e| matches!(e.stage, Stage::Vertex { .. }))
        .flat_map(|e| visit::reachable(m, e.function, &[]))
        .collect();
    if m.entry_points.iter().all(|e| vertex.contains(&e.function)) {
        return false;
    }
    let u32_ty = m.types.u32();
    let bool_ty = m.types.bool();
    let sum = m.add_resource(Resource {
        name: "wrela_tally".into(),
        binding: u32::MAX,
        kind: ResourceKind::Private,
        ty: u32_ty,
    });
    let flag = fuel::flag(m, binding);
    // Each statement's own work, from an unchanged copy of the module.
    let costs: Vec<Vec<u64>> = {
        let snapshot = m.clone();
        let mut w = ExprWork::new(&snapshot);
        snapshot
            .functions
            .iter()
            .map(|f| {
                let mut out = Vec::new();
                own_costs(&snapshot, &mut w, f, &f.body, &mut out);
                out
            })
            .collect()
    };
    for (i, cs) in costs.into_iter().enumerate() {
        if vertex.contains(&FuncId(i as u32)) {
            continue;
        }
        let mut f = std::mem::take(&mut m.functions[i]);
        let mut body = std::mem::take(&mut f.body);
        let mut t = Tally { f: &mut f, sum, u32_ty, costs: cs.into_iter() };
        t.block(&mut body);
        f.body = body;
        m.functions[i] = f;
    }
    let entries: Vec<FuncId> = m.entry_points.iter().map(|e| e.function).collect();
    for e in entries {
        if vertex.contains(&e) {
            continue;
        }
        let f = &mut m.functions[e.index()];
        let mut body = std::mem::take(&mut f.body);
        add_on_return(f, &mut body, sum, flag, code, u32_ty, bool_ty);
        if !matches!(body.last(), Some(Stmt::Return(_))) {
            let mut tail = Vec::new();
            add_sum(f, &mut tail, sum, flag, code, u32_ty, bool_ty);
            body.append(&mut tail);
        }
        let mut start = Vec::new();
        let zero = f.let_(&mut start, u32_ty, Expr::Const(Const::U32(0)));
        start.push(Stmt::Store(Place::root(PlaceRoot::Resource(sum)), zero));
        start.append(&mut body);
        f.body = start;
    }
    true
}

/// The own work of each statement of `b` and the blocks in it, in the order [`Tally::block`]
/// walks them: a `let` or an evaluation its expression's (a call none: the function adds its
/// own), a store its write's, a branch its test's, anything else none.
fn own_costs(m: &Module, w: &mut ExprWork<'_>, f: &Function, b: &[Stmt], out: &mut Vec<u64>) {
    for s in b {
        let c = match s {
            Stmt::Let(_, e) | Stmt::Eval(e) => match e {
                Expr::Call(..) => 0,
                _ => w.of(f, e).unwrap_or(0),
            },
            Stmt::Store(p, _) => store_cost(m, p),
            Stmt::If { .. } => work::OP,
            _ => 0,
        };
        out.push(c);
        match s {
            Stmt::If { then, else_, .. } => {
                own_costs(m, w, f, then, out);
                own_costs(m, w, f, else_, out);
            }
            Stmt::Loop { body, continuing } => {
                own_costs(m, w, f, body, out);
                own_costs(m, w, f, continuing, out);
            }
            _ => {}
        }
    }
}

/// A store's own cost, as the work count counts it: its index, and a write of storage or
/// workgroup memory.
fn store_cost(m: &Module, p: &Place) -> u64 {
    let index = if p.has_index() { work::OP } else { 0 };
    index
        + match &p.root {
            PlaceRoot::Resource(r) => match m.resources[r.index()].kind {
                ResourceKind::StorageReadWrite => work::MEMORY,
                ResourceKind::Workgroup => work::SHARED,
                _ => 0,
            },
            PlaceRoot::Ptr(_) | PlaceRoot::Param(_) => work::MEMORY,
            _ => 0,
        }
}

/// Writes the runs' additions into one function's body.
struct Tally<'a> {
    f: &'a mut Function,
    sum: ResourceId,
    u32_ty: TypeId,
    costs: std::vec::IntoIter<u64>,
}

impl Tally<'_> {
    /// `b` with each straight run's work added where the run ends.
    fn block(&mut self, b: &mut Block) {
        let old = std::mem::take(b);
        let mut run = 0u64;
        for mut s in old {
            let own = self.costs.next().unwrap_or(0);
            match &mut s {
                Stmt::If { then, else_, .. } => {
                    run = run.saturating_add(own);
                    self.flush(b, &mut run);
                    self.block(then);
                    self.block(else_);
                }
                Stmt::Loop { body, continuing } => {
                    self.flush(b, &mut run);
                    self.block(body);
                    self.block(continuing);
                }
                Stmt::Break | Stmt::Continue | Stmt::Return(_) | Stmt::Trap => {
                    self.flush(b, &mut run);
                }
                _ => run = run.saturating_add(own),
            }
            b.push(s);
        }
        self.flush(b, &mut run);
    }

    /// `sum += run` at the end of `b`, and the run begun again.
    fn flush(&mut self, b: &mut Block, run: &mut u64) {
        if *run == 0 {
            return;
        }
        let u = self.u32_ty;
        let at = Place::root(PlaceRoot::Resource(self.sum));
        let was = self.f.let_(b, u, Expr::Load(at.clone()));
        let add =
            self.f.let_(b, u, Expr::Const(Const::U32((*run).min(u64::from(u32::MAX)) as u32)));
        let now = self.f.let_(b, u, Expr::Binary(BinOp::Add, was, add));
        b.push(Stmt::Store(at, now));
        *run = 0;
    }
}

/// Before each `return` in `b` (an entry point's), [`add_sum`].
fn add_on_return(
    f: &mut Function,
    b: &mut Block,
    sum: ResourceId,
    flag: ResourceId,
    code: u32,
    u: TypeId,
    bt: TypeId,
) {
    let old = std::mem::take(b);
    for mut s in old {
        match &mut s {
            Stmt::If { then, else_, .. } => {
                add_on_return(f, then, sum, flag, code, u, bt);
                add_on_return(f, else_, sum, flag, code, u, bt);
            }
            Stmt::Loop { body, .. } => add_on_return(f, body, sum, flag, code, u, bt),
            Stmt::Return(_) => add_sum(f, b, sum, flag, code, u, bt),
            _ => {}
        }
        b.push(s);
    }
}

/// `old = atomicAdd(flag[w], sum); if old + sum < old { atomicAdd(flag[w + 1], 1) }`: the
/// invocation's sum into its pipeline's two words, the low one's carry into the high.
fn add_sum(
    f: &mut Function,
    out: &mut Block,
    sum: ResourceId,
    flag: ResourceId,
    code: u32,
    u: TypeId,
    bt: TypeId,
) {
    let s = f.let_(out, u, Expr::Load(Place::root(PlaceRoot::Resource(sum))));
    let lo = f.let_(out, u, Expr::Const(Const::U32(tally_word(code))));
    let at = Place { root: PlaceRoot::Resource(flag), path: vec![Proj::Index(lo)] };
    let old = f.let_(out, u, Expr::Atomic(AtomicOp::Add, at, vec![s]));
    let new = f.let_(out, u, Expr::Binary(BinOp::Add, old, s));
    let wrapped = f.let_(out, bt, Expr::Binary(BinOp::Lt, new, old));
    let mut then = Vec::new();
    let hi = f.let_(&mut then, u, Expr::Const(Const::U32(tally_word(code) + 1)));
    let one = f.let_(&mut then, u, Expr::Const(Const::U32(1)));
    let at_hi = Place { root: PlaceRoot::Resource(flag), path: vec![Proj::Index(hi)] };
    then.push(Stmt::Eval(Expr::Atomic(AtomicOp::Add, at_hi, vec![one])));
    out.push(Stmt::If { cond: wrapped, then, else_: Vec::new() });
}
