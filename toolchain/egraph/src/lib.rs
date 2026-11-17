//! A spike (wrela-notes/next/backend.md, "E-graph spike"): equality saturation (egg) over a
//! flattened GPU module's pure stretches, aware of rates and intervals.
//!
//! A **stretch** is a run of `let`s in one block with nothing between them that writes or
//! branches: pure computation, and reads (loads, texture reads, the invocation's inputs), which
//! no statement between them can change, so they may be computed in any order. Each stretch
//! becomes an e-graph: a let is a node, a value from before the stretch a leaf, a let the rules
//! can't see into (a read, a conversion) a node kept as written ([`lang::Node::Opaque`]). The
//! e-class analysis ([`analysis`]) carries each class's type, its rate (constant, the same for
//! the whole command, varying), an interval when one is known and whether it may be a NaN. After
//! saturation (with node, iteration and time limits) each value the rest of the function reads is
//! extracted at the least cost (§18.11's work, varying work at full price, uniform work at
//! [`cost::UNIFORM`]) and the stretch written again, its classes shared, in an order that
//! respects their operands. A stretch is written again only if that costs less than it did.

pub mod analysis;
pub mod cost;
pub mod hoist;
pub mod lang;
pub mod rules;
pub mod test_program;

use analysis::{Facts, Given, Graph, Rate, shape, type_of};
use egg::{Extractor, Id, Language, Rewrite, Runner, StopReason};
use lang::{Bop, Lit, Node, Uop};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};
use wrela_ir::{Expr, Function, Module, Stmt, TypeDef, TypeId, Types, ValueId};

/// What the pass may do.
#[derive(Clone, Debug)]
pub struct Options {
    pub groups: rules::Groups,
    pub node_limit: usize,
    pub iter_limit: usize,
    /// For each stretch's saturation.
    pub time_limit: Duration,
    /// Move uniform work the host can compute out of the shader (`WRELA_EGG_HOIST`).
    pub hoist: bool,
}

impl Options {
    /// From `WRELA_EGG`: `exact` (exact and interval rules), `approx` (and the approximate ones)
    /// or `reassoc` (and reassociation); unset or empty, no pass. `WRELA_EGG_NODES`,
    /// `WRELA_EGG_ITERS` and `WRELA_EGG_MS` set the limits.
    pub fn from_env() -> Option<Options> {
        let groups = match std::env::var("WRELA_EGG").ok()?.as_str() {
            "exact" => rules::Groups { approx: false, reassoc: false },
            "approx" => rules::Groups { approx: true, reassoc: false },
            "reassoc" => rules::Groups { approx: true, reassoc: true },
            _ => return None,
        };
        let num =
            |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        Some(Options {
            groups,
            node_limit: num("WRELA_EGG_NODES", 20_000),
            iter_limit: num("WRELA_EGG_ITERS", 12),
            time_limit: Duration::from_millis(num("WRELA_EGG_MS", 100) as u64),
            hoist: std::env::var_os("WRELA_EGG_HOIST").is_some(),
        })
    }
}

/// What the pass did to a module.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub stretches: usize,
    /// Written again (cheaper).
    pub rewritten: usize,
    /// Stretches whose saturation stopped at a limit, not saturated.
    pub limited: usize,
    pub lets_before: usize,
    pub lets_after: usize,
    /// §18.11 work, counted once a statement (not by how often it runs), of the values that vary
    /// between invocations and of those the same for the whole command (constants not counted).
    pub varying_before: f64,
    pub varying_after: f64,
    pub uniform_before: f64,
    pub uniform_after: f64,
    pub nodes: usize,
    pub seconds: f64,
    /// Matches each rule applied (that changed the e-graph), summed over the stretches.
    pub applied: BTreeMap<String, usize>,
    /// Hoisting: the work moved out of the shader (§18.11, counted once at each read), the
    /// reads of hoisted fields written, the fields, and the tape that fills them.
    pub hoisted_work: f64,
    pub hoisted_reads: usize,
    pub hoisted_fields: usize,
    pub tape: Option<wrela_abi::hoist::Tape>,
}

impl Stats {
    pub fn add(&mut self, o: &Stats) {
        self.stretches += o.stretches;
        self.rewritten += o.rewritten;
        self.limited += o.limited;
        self.lets_before += o.lets_before;
        self.lets_after += o.lets_after;
        self.varying_before += o.varying_before;
        self.varying_after += o.varying_after;
        self.uniform_before += o.uniform_before;
        self.uniform_after += o.uniform_after;
        self.nodes += o.nodes;
        self.seconds += o.seconds;
        for (k, v) in &o.applied {
            *self.applied.entry(k.clone()).or_default() += v;
        }
    }

    /// One line, and the rules that applied most.
    pub fn summary(&self) -> String {
        let mut rules: Vec<_> = self.applied.iter().collect();
        rules.sort_by(|a, b| b.1.cmp(a.1));
        let top: Vec<String> = rules.iter().take(12).map(|(k, v)| format!("{k} {v}")).collect();
        format!(
            "{} stretches, {} rewritten, {} at a limit; lets {} -> {}; varying work {:.0} -> {:.0}; \
             uniform work {:.0} -> {:.0}; hoisted {:.0} work in {} reads of {} fields; {} e-nodes; {:.1} ms; applied: {}",
            self.stretches,
            self.rewritten,
            self.limited,
            self.lets_before,
            self.lets_after,
            self.varying_before,
            self.varying_after,
            self.uniform_before,
            self.uniform_after,
            self.hoisted_work,
            self.hoisted_reads,
            self.hoisted_fields,
            self.nodes,
            self.seconds * 1000.0,
            top.join(", ")
        )
    }
}

/// Rewrites a flattened GPU module's stretches.
pub fn rewrite_module(m: &mut Module, opts: &Options) -> Stats {
    let start = Instant::now();
    let rewrites = rules::rewrites(opts.groups);
    let mut stats = Stats::default();
    // What needs the whole module, first: which values are the same for every invocation, and
    // each let's work.
    let mut facts: Vec<(HashSet<ValueId>, HashMap<ValueId, f64>)> = Vec::new();
    for f in &m.functions {
        let args = vec![false; f.params.len()];
        let same = wrela_ir::same::same_values(m, f, &args);
        let mut work = HashMap::new();
        let mut w = wrela_ir::work::ExprWork::new(m);
        wrela_ir::visit::walk(&f.body, &mut |s| {
            if let Stmt::Let(v, e) = s {
                work.insert(*v, w.of(f, e).unwrap_or(0) as f64);
            }
        });
        facts.push((same, work));
    }
    let fake = std::env::var_os("WRELA_EGG_FAKE_HOIST").and_then(|_| uniform_f32(m));
    if let Some(p) = &fake {
        FAKE_SCALAR.store(
            m.types.lookup(&TypeDef::Scalar(wrela_ir::Scalar::F32)).map_or(0, |t| t.0),
            std::sync::atomic::Ordering::Relaxed,
        );
        let _ = p;
    }
    let mut hoister = if opts.hoist { hoist::Hoist::new(m) } else { None };
    let mut types = std::mem::take(&mut m.types);
    for (i, (same, work)) in facts.into_iter().enumerate() {
        let mut f = std::mem::take(&mut m.functions[i]);
        let mut cx = Cx {
            types: &mut types,
            same: &same,
            work: &work,
            rewrites: &rewrites,
            opts,
            stats: &mut stats,
            fake: fake.clone(),
            hoist: hoister.as_mut(),
        };
        cx.function(&mut f);
        m.functions[i] = f;
    }
    m.types = types;
    if let Some(h) = hoister {
        stats.hoisted_work = h.moved_work;
        stats.hoisted_reads = h.reads;
        stats.hoisted_fields = h.fields.len();
        stats.tape = h.finish(m);
    }
    stats.seconds = start.elapsed().as_secs_f64();
    stats
}

struct Cx<'a> {
    types: &'a mut Types,
    same: &'a HashSet<ValueId>,
    work: &'a HashMap<ValueId, f64>,
    rewrites: &'a [Rewrite<Node, Facts>],
    opts: &'a Options,
    stats: &'a mut Stats,
    fake: Option<wrela_ir::Place>,
    hoist: Option<&'a mut hoist::Hoist>,
}

/// Whether a statement can be part of a stretch: a `let` that computes or reads.
fn in_stretch(s: &Stmt) -> bool {
    match s {
        Stmt::At(_) => true,
        Stmt::Let(_, e) => {
            e.only_computes()
                || matches!(
                    e,
                    Expr::Load(_)
                        | Expr::EntryInput(_)
                        | Expr::Param(_)
                        | Expr::Texture(..)
                        | Expr::ArrayLength(_)
                )
        }
        _ => false,
    }
}

impl Cx<'_> {
    fn function(&mut self, f: &mut Function) {
        let mut consts = HashMap::new();
        let mut defs = HashMap::new();
        let mut uses: HashMap<ValueId, u32> = HashMap::new();
        wrela_ir::visit::walk(&f.body, &mut |s| {
            if let Stmt::Let(v, e) = s {
                defs.insert(*v, e.clone());
            }
            if let Stmt::Let(v, Expr::Const(c)) = s
                && let Some(l) = Lit::from_const(c)
            {
                consts.insert(*v, l);
            }
            s.for_each_value(&mut |v| *uses.entry(v).or_default() += 1);
        });
        let mut st = FnState { consts, defs, uses, rename: HashMap::new() };
        let mut body = std::mem::take(&mut f.body);
        self.block(f, &mut st, &mut body);
        f.body = body;
        if st.rename.is_empty() {
            return;
        }
        let resolve = |mut v: ValueId| {
            while let Some(&n) = st.rename.get(&v) {
                v = n;
            }
            v
        };
        wrela_ir::visit::walk_mut(&mut f.body, &mut |s| {
            s.for_each_value_mut(&mut |v| *v = resolve(*v))
        });
    }

    fn block(&mut self, f: &mut Function, st: &mut FnState, block: &mut Vec<Stmt>) {
        let old = std::mem::take(block);
        let mut run = Vec::new();
        for mut s in old {
            if in_stretch(&s) {
                run.push(s);
                continue;
            }
            self.stretch(f, st, std::mem::take(&mut run), block);
            for b in s.blocks_mut() {
                self.block(f, st, b);
            }
            block.push(s);
        }
        self.stretch(f, st, run, block);
    }

    /// Rewrites one stretch into `out`.
    fn stretch(&mut self, f: &mut Function, st: &mut FnState, run: Vec<Stmt>, out: &mut Vec<Stmt>) {
        let lets: Vec<(ValueId, Expr)> = run
            .iter()
            .filter_map(|s| match s {
                Stmt::Let(v, e) => Some((*v, e.clone())),
                _ => None,
            })
            .collect();
        if lets.len() < 2 {
            out.extend(run);
            return;
        }
        let mut inside: HashMap<ValueId, u32> = HashMap::new();
        for (_, e) in &lets {
            e.for_each_value(&mut |v| *inside.entry(v).or_default() += 1);
        }
        let roots: Vec<ValueId> = lets
            .iter()
            .map(|(v, _)| *v)
            .filter(|v| st.uses.get(v).copied().unwrap_or(0) > inside.get(v).copied().unwrap_or(0))
            .collect();
        if roots.is_empty() {
            out.extend(run);
            return;
        }
        self.stats.stretches += 1;

        // The e-graph of the stretch as written.
        let types = std::mem::take(self.types);
        let mut eg: Graph = egg::EGraph::new(Facts { types, given: HashMap::new() });
        let mut class: HashMap<ValueId, Id> = HashMap::new();
        let mut written: HashMap<Id, Node> = HashMap::new();
        let mut exprs: HashMap<u32, Expr> = HashMap::new();
        for (v, e) in &lets {
            let mut kids = Vec::new();
            e.for_each_value(&mut |x| kids.push(x));
            let ids: Vec<Id> =
                kids.iter().map(|&x| self.operand(f, st, &mut eg, &class, x)).collect();
            let ty = f.value_ty(*v);
            let node = self.node(&mut eg, e, &ids, &kids, ty).unwrap_or_else(|| {
                let rate = if self.same.contains(v) { Rate::Uniform } else { Rate::Varying };
                let work = self.work.get(v).copied().unwrap_or(0.0);
                eg.analysis.given.insert(v.0, Given { ty, rate, work });
                exprs.insert(v.0, e.clone());
                Node::Opaque(v.0, ids.into_boxed_slice())
            });
            let id = eg.add(node.clone());
            written.entry(id).or_insert(node);
            class.insert(*v, id);
        }
        let root_classes: Vec<Id> = roots.iter().map(|v| class[v]).collect();
        if std::env::var_os("WRELA_EGG_DEBUG").is_some() {
            for (v, e) in &lets {
                let c = eg.find(class[v]);
                eprintln!(
                    "egg-debug: v{} = {e:?}  class {c} {:?} node {:?}",
                    v.0, eg[c].data, eg[c].nodes
                );
            }
            eprintln!("egg-debug: ---- roots {roots:?}");
        }
        // As written: each let's own node; a literal or leaf class its one node.
        let written: HashMap<Id, Node> =
            written.into_iter().map(|(k, n)| (eg.find(k), n)).collect();
        let before = dag(&eg, &root_classes, |c| {
            written.get(&c).cloned().unwrap_or_else(|| eg[c].nodes[0].clone())
        });

        let runner = Runner::default()
            .with_egraph(eg)
            .with_node_limit(self.opts.node_limit)
            .with_iter_limit(self.opts.iter_limit)
            .with_time_limit(self.opts.time_limit)
            .run(self.rewrites);
        let stop = runner.stop_reason.clone();
        let runner_iterations = runner.iterations.clone();
        if !matches!(runner.stop_reason, Some(StopReason::Saturated)) {
            self.stats.limited += 1;
        }
        for it in &runner.iterations {
            for (k, n) in &it.applied {
                *self.stats.applied.entry(k.to_string()).or_default() += n;
            }
        }
        let mut eg = runner.egraph;
        self.stats.nodes += eg.total_number_of_nodes();
        let best: HashMap<Id, Node> = {
            let ex = Extractor::new(&eg, cost::Cost(&eg));
            let mut best = HashMap::new();
            let mut stack: Vec<Id> = root_classes.iter().map(|&c| eg.find(c)).collect();
            while let Some(c) = stack.pop() {
                if best.contains_key(&c) {
                    continue;
                }
                let n = ex.find_best_node(c).clone();
                stack.extend(n.children().iter().map(|&k| eg.find(k)));
                best.insert(c, n);
            }
            best
        };
        let roots_now: Vec<Id> = root_classes.iter().map(|&c| eg.find(c)).collect();
        let after = dag(&eg, &roots_now, |c| best[&eg.find(c)].clone());
        if std::env::var_os("WRELA_EGG_DEBUG").is_some() {
            let mut applied: BTreeMap<String, usize> = BTreeMap::new();
            for it in &runner_iterations {
                for (k, n) in &it.applied {
                    *applied.entry(k.to_string()).or_default() += n;
                }
            }
            eprintln!(
                "egg-debug: ==== {:?} cost {} -> {}; applied {applied:?}",
                stop, before.cost, after.cost
            );
        }
        self.stats.lets_before += lets.len();
        self.stats.varying_before += before.varying;
        self.stats.uniform_before += before.uniform;
        if after.cost + 1e-9 >= before.cost && self.fake.is_none() && self.hoist.is_none() {
            *self.types = std::mem::take(&mut eg.analysis.types);
            self.stats.lets_after += lets.len();
            self.stats.varying_after += before.varying;
            self.stats.uniform_after += before.uniform;
            out.extend(run);
            return;
        }
        self.stats.rewritten += 1;
        self.stats.varying_after += after.varying;
        self.stats.uniform_after += after.uniform;
        // The stretch written again: its source lines first, then each class it needs, after its
        // operands.
        out.extend(run.into_iter().filter(|s| matches!(s, Stmt::At(_))));
        let n0 = out.len();
        {
            let mut em = Emit {
                eg: &eg,
                best: &best,
                exprs: &exprs,
                emitted: HashMap::new(),
                fake: self.fake.as_ref(),
                hoist: self.hoist.as_deref_mut(),
                defs: &st.defs,
                tvs: HashMap::new(),
                work: HashMap::new(),
            };
            for (v, &c) in roots.iter().zip(&roots_now) {
                let nv = em.go(f, out, c);
                st.rename.insert(*v, nv);
            }
        }
        self.stats.lets_after += out.len() - n0;
        *self.types = std::mem::take(&mut eg.analysis.types);
    }

    /// The class of an operand: a value of the stretch, a constant's literal, or a leaf.
    fn operand(
        &self,
        f: &Function,
        st: &FnState,
        eg: &mut Graph,
        class: &HashMap<ValueId, Id>,
        x: ValueId,
    ) -> Id {
        if let Some(&c) = class.get(&x) {
            return c;
        }
        if let Some(&l) = st.consts.get(&x) {
            return eg.add(Node::Lit(l));
        }
        let rate = if self.same.contains(&x) { Rate::Uniform } else { Rate::Varying };
        eg.analysis.given.insert(x.0, Given { ty: f.value_ty(x), rate, work: 0.0 });
        eg.add(Node::Leaf(x.0))
    }

    /// A let as a node the rules can see into, if it is one (its type checked).
    fn node(
        &self,
        eg: &mut Graph,
        e: &Expr,
        ids: &[Id],
        kids: &[ValueId],
        ty: TypeId,
    ) -> Option<Node> {
        let one = |i: usize| [ids[i]];
        let node = match e {
            Expr::Const(c) => Node::Lit(Lit::from_const(c)?),
            Expr::Unary(op, _) => Node::Un(Uop::of(*op), one(0)),
            Expr::Binary(op, _, _) => Node::Bin(Bop::of(*op), [ids[0], ids[1]]),
            Expr::Builtin(b, _) => Node::Call(*b, ids.into()),
            Expr::Splat(_, n) => Node::Splat(*n, one(0)),
            Expr::Swizzle(_, idx) => Node::Swizzle(idx.clone().into_boxed_slice(), one(0)),
            Expr::Extract(_, k) => Node::Extract(*k, one(0)),
            Expr::Construct(t, _) => {
                // `vec3 { x, x, x }` is a splat.
                let splat = match eg.analysis.types.get(*t) {
                    TypeDef::Vector(_, n) => {
                        usize::from(*n) == kids.len() && kids.iter().all(|&k| k == kids[0])
                    }
                    _ => false,
                };
                let scalar = splat
                    && shape(&eg.analysis.types, eg[ids[0]].data.ty).is_some_and(|(_, w)| w == 1);
                if scalar {
                    let TypeDef::Vector(_, n) = *eg.analysis.types.get(*t) else { unreachable!() };
                    Node::Splat(n, one(0))
                } else {
                    Node::Construct(t.0, ids.into())
                }
            }
            Expr::Select { .. } => Node::Select([ids[0], ids[1], ids[2]]),
            _ => return None,
        };
        // Only scalars and vectors are rewritten (a matrix product stays as written).
        let shaped =
            node.children().iter().all(|&c| shape(&eg.analysis.types, eg[c].data.ty).is_some())
                || matches!(node, Node::Extract(..) | Node::Construct(..));
        let kid_tys: Vec<TypeId> = node.children().iter().map(|&c| eg[c].data.ty).collect();
        let Facts { types, given } = &mut eg.analysis;
        (shaped && type_of(types, given, &node, &kid_tys) == Some(ty)).then_some(node)
    }
}

/// A function's state through the pass.
struct FnState {
    consts: HashMap<ValueId, Lit>,
    /// Every let's expression as written (for hoisting a leaf's value).
    defs: HashMap<ValueId, Expr>,
    uses: HashMap<ValueId, u32>,
    /// A stretch's values the rest of the function reads, to what now computes them.
    rename: HashMap<ValueId, ValueId>,
}

struct Dag {
    cost: f64,
    varying: f64,
    uniform: f64,
}

/// The cost of computing `roots` with each class's node from `pick`, each class once.
fn dag(eg: &Graph, roots: &[Id], pick: impl Fn(Id) -> Node) -> Dag {
    let mut d = Dag { cost: 0.0, varying: 0.0, uniform: 0.0 };
    let mut seen = HashSet::new();
    let mut stack: Vec<Id> = roots.to_vec();
    while let Some(c) = stack.pop() {
        let c = eg.find(c);
        if !seen.insert(c) {
            continue;
        }
        let n = pick(c);
        d.cost += cost::own_cost(eg, &n);
        let w = cost::own_work(eg, &n);
        match cost::node_rate(eg, &n) {
            Rate::Varying => d.varying += w,
            Rate::Uniform => d.uniform += w,
            Rate::Const => {}
        }
        stack.extend(n.children().iter().copied());
    }
    d
}

static FAKE_SCALAR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn fake_scalar_ty() -> u32 {
    FAKE_SCALAR.load(std::sync::atomic::Ordering::Relaxed)
}

/// The first `f32` of the module's uniform block, as a place (for the hoisting ceiling).
fn uniform_f32(m: &Module) -> Option<wrela_ir::Place> {
    use wrela_ir::{Place, PlaceRoot, Proj, ResourceId, ResourceKind};
    let (i, r) = m
        .resources
        .iter()
        .enumerate()
        .find(|(_, r)| matches!(r.kind, ResourceKind::Uniform { .. }))?;
    fn find(m: &Module, t: TypeId, path: &mut Vec<Proj>) -> bool {
        match m.types.get(t) {
            TypeDef::Scalar(wrela_ir::Scalar::F32) => true,
            TypeDef::Struct { .. } => {
                let mut k = 0;
                while let Some(ft) = m.types.field(t, k) {
                    path.push(Proj::Field(k));
                    if find(m, ft, path) {
                        return true;
                    }
                    path.pop();
                    k += 1;
                }
                false
            }
            _ => false,
        }
    }
    let mut path = Vec::new();
    find(m, r.ty, &mut path)
        .then_some(Place { root: PlaceRoot::Resource(ResourceId(i as u32)), path })
}

/// Writes a stretch's classes as lets, each once, after its operands.
struct Emit<'a> {
    eg: &'a Graph,
    best: &'a HashMap<Id, Node>,
    exprs: &'a HashMap<u32, Expr>,
    emitted: HashMap<Id, ValueId>,
    fake: Option<&'a wrela_ir::Place>,
    hoist: Option<&'a mut hoist::Hoist>,
    defs: &'a HashMap<ValueId, Expr>,
    tvs: HashMap<Id, Option<hoist::Tv>>,
    work: HashMap<Id, f64>,
}

impl Emit<'_> {
    /// The work of a class's chosen form, each class of it once (leaves are free here).
    fn subtree_work(&mut self, c: Id, seen: &mut std::collections::HashSet<Id>) -> f64 {
        let c = self.eg.find(c);
        if !seen.insert(c) {
            return 0.0;
        }
        let n = &self.best[&c];
        let own = cost::own_work(self.eg, n);
        let kids: Vec<Id> = n.children().to_vec();
        own + kids.into_iter().map(|k| self.subtree_work(k, seen)).sum::<f64>()
    }

    /// The class as a value the host can compute, if it is one.
    fn tv(&mut self, f: &Function, c: Id) -> Option<hoist::Tv> {
        let c = self.eg.find(c);
        if let Some(t) = self.tvs.get(&c) {
            return t.clone();
        }
        // Guard against a cycle through this class while it's being worked out.
        self.tvs.insert(c, None);
        let t = self.tv_of(f, c);
        self.tvs.insert(c, t.clone());
        t
    }

    fn tv_of(&mut self, f: &Function, c: Id) -> Option<hoist::Tv> {
        use hoist::Term;
        let n = self.best[&c].clone();
        let ty = self.eg[c].data.ty;
        let types = &self.eg.analysis.types;
        let kids = n.children().to_vec();
        let mut k = Vec::new();
        if !matches!(n, Node::Opaque(..)) {
            for &x in &kids {
                k.push(self.tv(f, x)?);
            }
        }
        let term = match &n {
            Node::Leaf(x) => {
                let defs = self.defs;
                return self.hoist.as_deref_mut()?.value(types, defs, f, ValueId(*x), 0);
            }
            Node::Lit(l) => Term::Lit(l.to_const()),
            Node::Bin(op, _) => Term::Bin(op.op(), k[0].clone(), k[1].clone()),
            Node::Un(op, _) => Term::Un(op.op(), k[0].clone()),
            Node::Call(b, _) => Term::Call(*b, k),
            Node::Splat(w, _) => Term::Splat(*w, k[0].clone()),
            Node::Swizzle(idx, _) => Term::Swizzle(idx.to_vec(), k[0].clone()),
            Node::Extract(i, _) => Term::Extract(*i, k[0].clone()),
            Node::Construct(t, _) => Term::Construct(TypeId(*t), k),
            Node::Select(_) => Term::Select(k[0].clone(), k[1].clone(), k[2].clone()),
            Node::Opaque(x, _) => match &self.exprs[x] {
                Expr::Load(p) => {
                    // Its indices are its operands: each must be a constant.
                    let mut idx = Vec::new();
                    for &kid in &kids {
                        let lit = self.eg[kid].nodes.iter().find_map(|n| match n {
                            Node::Lit(l) => match l.to_const() {
                                wrela_ir::Const::U32(i) => Some(i),
                                wrela_ir::Const::I32(i) if i >= 0 => Some(i as u32),
                                _ => None,
                            },
                            _ => None,
                        });
                        idx.push(lit?);
                    }
                    if p.root != wrela_ir::PlaceRoot::Resource(self.hoist.as_deref()?.res) {
                        return None;
                    }
                    Term::Load(p.clone(), idx)
                }
                Expr::Convert(_, s) => Term::Convert(self.tv(f, kids[0])?, *s),
                _ => return None,
            },
        };
        self.hoist.as_deref_mut()?.term(types, term, ty)
    }

    /// The class's value, writing it (and what it needs) if it isn't yet.
    fn go(&mut self, f: &mut Function, out: &mut Vec<Stmt>, c: Id) -> ValueId {
        let eg = self.eg;
        let c = eg.find(c);
        if let Some(&v) = self.emitted.get(&c) {
            return v;
        }
        let n = self.best[&c].clone();
        // The hoisting ceiling (`WRELA_EGG_FAKE_HOIST`): a float value computed from uniform
        // values is read from the uniform block instead, as if computed once for the command.
        // Wrong pictures; only the time means anything.
        if let Some(place) = self.fake
            && eg[c].data.rate == Rate::Uniform
            && matches!(n, Node::Bin(..) | Node::Un(..) | Node::Call(..) | Node::Select(..))
            && let Some((wrela_ir::Scalar::F32, w)) = shape(&eg.analysis.types, eg[c].data.ty)
        {
            let x = f.new_value(TypeId(fake_scalar_ty()));
            out.push(Stmt::Let(x, Expr::Load(place.clone())));
            let v = if w == 1 {
                x
            } else {
                let v = f.new_value(eg[c].data.ty);
                out.push(Stmt::Let(v, Expr::Splat(x, w)));
                v
            };
            self.emitted.insert(c, v);
            return v;
        }
        // Hoisting (`WRELA_EGG_HOIST`): uniform work the host can compute is read from the
        // uniform block, computed once a command.
        if self.hoist.is_some()
            && eg[c].data.rate == Rate::Uniform
            && matches!(
                n,
                Node::Bin(..)
                    | Node::Un(..)
                    | Node::Call(..)
                    | Node::Select(..)
                    | Node::Construct(..)
                    | Node::Opaque(..)
            )
        {
            let w = match self.work.get(&c) {
                Some(&w) => w,
                None => {
                    let w = self.subtree_work(c, &mut HashSet::new());
                    self.work.insert(c, w);
                    w
                }
            };
            if w >= 2.0
                && let Some(t) = self.tv(f, c)
                && let Some((field, boolean)) = {
                    let types = &eg.analysis.types;
                    self.hoist.as_deref_mut().and_then(|h| h.slot(types, &t))
                }
            {
                let h = self.hoist.as_deref_mut().expect("hoisting");
                h.moved_work += w;
                h.reads += 1;
                let place = h.place(field);
                let v = if boolean {
                    let u = eg
                        .analysis
                        .types
                        .lookup(&TypeDef::Scalar(wrela_ir::Scalar::U32))
                        .expect("u32");
                    let x = f.new_value(u);
                    out.push(Stmt::Let(x, Expr::Load(place)));
                    let zero = f.new_value(u);
                    out.push(Stmt::Let(zero, Expr::Const(wrela_ir::Const::U32(0))));
                    let v = f.new_value(eg[c].data.ty);
                    out.push(Stmt::Let(v, Expr::Binary(wrela_ir::BinOp::Ne, x, zero)));
                    v
                } else {
                    let v = f.new_value(eg[c].data.ty);
                    out.push(Stmt::Let(v, Expr::Load(place)));
                    v
                };
                self.emitted.insert(c, v);
                return v;
            }
        }
        let kids: Vec<ValueId> = n.children().iter().map(|&k| self.go(f, out, k)).collect();
        let e = match &n {
            Node::Leaf(x) => {
                self.emitted.insert(c, ValueId(*x));
                return ValueId(*x);
            }
            Node::Lit(l) => Expr::Const(l.to_const()),
            Node::Bin(op, _) => Expr::Binary(op.op(), kids[0], kids[1]),
            Node::Un(op, _) => Expr::Unary(op.op(), kids[0]),
            Node::Call(b, _) => Expr::Builtin(*b, kids),
            Node::Splat(w, _) => Expr::Splat(kids[0], *w),
            Node::Swizzle(idx, _) => Expr::Swizzle(kids[0], idx.to_vec()),
            Node::Extract(k, _) => Expr::Extract(kids[0], *k),
            Node::Construct(t, _) => Expr::Construct(TypeId(*t), kids),
            Node::Select(_) => Expr::Select { cond: kids[0], if_true: kids[1], if_false: kids[2] },
            Node::Opaque(x, _) => {
                let mut e = self.exprs[x].clone();
                let mut i = 0;
                e.for_each_value_mut(&mut |v| {
                    *v = kids[i];
                    i += 1;
                });
                e
            }
        };
        let v = f.new_value(eg[c].data.ty);
        out.push(Stmt::Let(v, e));
        self.emitted.insert(c, v);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::RecExpr;
    use wrela_ir::Scalar;

    /// An e-graph of `expr`, its leaves `v0..` given as (scalar, width, rate), saturated with
    /// `groups`; the best form of its root and its cost before and after.
    pub(crate) fn best(
        expr: &str,
        leaves: &[(Scalar, u8, Rate)],
        groups: rules::Groups,
    ) -> (String, f64, f64) {
        let mut types = Types::default();
        let mut given = HashMap::new();
        for (i, &(s, n, rate)) in leaves.iter().enumerate() {
            let ty = if n == 1 { types.scalar(s) } else { types.vector_of(s, n) };
            given.insert(i as u32, Given { ty, rate, work: 0.0 });
        }
        let mut eg: Graph = egg::EGraph::new(Facts { types, given });
        let e: RecExpr<Node> = expr.parse().expect("parses");
        let root = eg.add_expr(&e);
        eg.rebuild();
        let before = Extractor::new(&eg, cost::Cost(&eg)).find_best(root).0;
        let opts = Options {
            groups,
            node_limit: 20_000,
            iter_limit: 12,
            time_limit: Duration::from_secs(1),
            hoist: false,
        };
        let runner = Runner::default()
            .with_egraph(eg)
            .with_node_limit(opts.node_limit)
            .run(&rules::rewrites(groups));
        let eg = runner.egraph;
        let (after, e) = Extractor::new(&eg, cost::Cost(&eg)).find_best(root);
        (e.to_string(), before, after)
    }

    const APPROX: rules::Groups = rules::Groups { approx: true, reassoc: false };
    const ALL: rules::Groups = rules::Groups { approx: true, reassoc: true };
    const V: Rate = Rate::Varying;
    const U: Rate = Rate::Uniform;

    #[test]
    fn a_log_of_a_quotient_by_a_constant_splits() {
        let (e, b, a) = best(
            "(log2 (/ (* (max (length v0) (length v1)) 256) 19.7))",
            &[(Scalar::F32, 2, V), (Scalar::F32, 2, V)],
            APPROX,
        );
        assert!(a < b, "{e}: {b} -> {a}");
    }

    #[test]
    fn a_uniform_divisor_becomes_a_reciprocal() {
        let (e, b, a) = best("(/ v0 v1)", &[(Scalar::F32, 1, V), (Scalar::F32, 1, U)], APPROX);
        assert!(a < b, "{e}: {b} -> {a}");
        assert_eq!(e, "(* v0 (/ 1.0 v1))");
    }

    #[test]
    fn uniform_factors_gather() {
        let (e, b, a) = best(
            "(* (* v0 v1) v2)",
            &[(Scalar::F32, 1, V), (Scalar::F32, 1, U), (Scalar::F32, 1, U)],
            ALL,
        );
        assert!(a < b, "{e}: {b} -> {a}");
    }

    #[test]
    fn an_abs_of_a_square_root_goes() {
        let (e, _, _) = best("(abs (sqrt v0))", &[(Scalar::F32, 1, V)], APPROX);
        assert_eq!(e, "(sqrt v0)");
    }

    #[test]
    fn every_rule_parses() {
        assert!(rules::rewrites(ALL).len() > 50);
    }
}

/// For the spike's ceilings: of a fragment entry point's function, the static §18.11 work of
/// its lets, of those computed only from its varyings, uniform values and constants by pure
/// operations (movable to the vertex shader, exactly where affine), and of those the same for
/// the command. Locals count as varying.
pub fn fragment_rates(m: &Module) -> Vec<(String, f64, f64, f64)> {
    let mut out = Vec::new();
    for e in &m.entry_points {
        if !matches!(e.stage, wrela_ir::Stage::Fragment { .. }) {
            continue;
        }
        let f = &m.functions[e.function.index()];
        let same = wrela_ir::same::same_values(m, f, &vec![false; f.params.len()]);
        let mut w = wrela_ir::work::ExprWork::new(m);
        let mut interp: HashSet<ValueId> = HashSet::new();
        let (mut total, mut inter, mut uni) = (0.0, 0.0, 0.0);
        wrela_ir::visit::walk(&f.body, &mut |s| {
            if let Stmt::Let(v, ex) = s {
                let work = w.of(f, ex).unwrap_or(0) as f64;
                total += work;
                if same.contains(v) {
                    uni += work;
                    return;
                }
                let mut ok = ex.only_computes() || matches!(ex, Expr::Param(_));
                ex.for_each_value(&mut |x| ok &= interp.contains(&x) || same.contains(&x));
                if ok
                    && !matches!(
                        ex,
                        Expr::Builtin(
                            wrela_ir::Builtin::Dpdx
                                | wrela_ir::Builtin::Dpdy
                                | wrela_ir::Builtin::Fwidth,
                            _
                        )
                    )
                {
                    interp.insert(*v);
                    inter += work;
                }
            }
        });
        let half = half_sinks(m, f);
        let mut w2 = wrela_ir::work::ExprWork::new(m);
        let mut h = 0.0;
        wrela_ir::visit::walk(&f.body, &mut |s| {
            if let Stmt::Let(v, ex) = s
                && half.contains(v)
            {
                h += w2.of(f, ex).unwrap_or(0) as f64;
            }
        });
        let reach = reach_colour(m, f);
        let mut w3 = wrela_ir::work::ExprWork::new(m);
        let mut rc = 0.0;
        let mut alu = 0.0;
        wrela_ir::visit::walk(&f.body, &mut |s| {
            if let Stmt::Let(v, ex) = s {
                let w = w3.of(f, ex).unwrap_or(0) as f64;
                if ex.only_computes() {
                    alu += w;
                }
                if reach.contains(v) {
                    rc += w;
                }
            }
        });
        out.push((
            format!("{} (half-eligible {h:.0}, reaching colour {rc:.0} of ALU {alu:.0})", e.name),
            total,
            inter,
            uni,
        ));
    }
    out
}

/// The values of a fragment function that may be computed in half precision: `f32` scalars and
/// vectors computed by pure arithmetic whose every use, transitively, is arithmetic of the same
/// kind or the colour of output 0 (through locals too), never a comparison, a branch, a texture
/// coordinate, an index or another output. Optimistic start, removed until stable.
pub fn half_sinks(m: &Module, f: &Function) -> HashSet<ValueId> {
    use wrela_ir::{Builtin as B, Scalar};
    let float = |t: TypeId| shape(&m.types, t).is_some_and(|(s, _)| s == Scalar::F32);
    let mut lets: HashMap<ValueId, &Expr> = HashMap::new();
    let mut loads: HashMap<wrela_ir::LocalId, Vec<ValueId>> = HashMap::new();
    let mut stores: Vec<(wrela_ir::LocalId, ValueId)> = Vec::new();
    let mut ret: Option<ValueId> = None;
    wrela_ir::visit::walk(&f.body, &mut |s| match s {
        Stmt::Let(v, e) => {
            lets.insert(*v, e);
            if let Expr::Load(p) = e
                && p.path.is_empty()
                && let Some(l) = p.root_local()
            {
                loads.entry(l).or_default().push(*v);
            }
        }
        Stmt::Store(p, v) if p.path.is_empty() && p.root_local().is_some() => {
            stores.push((p.root_local().unwrap(), *v))
        }
        Stmt::Return(Some(v)) => ret = Some(*v),
        _ => {}
    });
    // The colour: output 0 of the returned targets.
    let colour = ret.and_then(|r| match lets.get(&r) {
        Some(Expr::Construct(_, xs)) => xs.first().copied(),
        _ => None,
    });
    let pure = |e: &Expr| match e {
        Expr::Binary(op, ..) => matches!(
            op,
            wrela_ir::BinOp::Add
                | wrela_ir::BinOp::Sub
                | wrela_ir::BinOp::Mul
                | wrela_ir::BinOp::Div
        ),
        Expr::Unary(wrela_ir::UnOp::Neg, _) => true,
        Expr::Builtin(b, _) => !matches!(b, B::Dpdx | B::Dpdy | B::Fwidth | B::AllEqual),
        Expr::Construct(..)
        | Expr::Splat(..)
        | Expr::Swizzle(..)
        | Expr::Extract(..)
        | Expr::Const(_) => true,
        Expr::Select { .. } => true,
        Expr::Load(_) => true,
        _ => false,
    };
    let mut half: HashSet<ValueId> =
        lets.iter().filter(|(v, e)| float(f.value_ty(**v)) && pure(e)).map(|(v, _)| *v).collect();
    let mut half_locals: HashSet<wrela_ir::LocalId> =
        loads.keys().copied().filter(|l| float(f.locals[l.index()].ty)).collect();
    loop {
        // Every use of each value.
        let mut ok: HashMap<ValueId, bool> = HashMap::new();
        let mut mark = |v: ValueId, good: bool| {
            let e = ok.entry(v).or_insert(true);
            *e &= good;
        };
        wrela_ir::visit::walk(&f.body, &mut |s| match s {
            Stmt::Let(d, e) => {
                let good_user = half.contains(d) && !matches!(e, Expr::Load(_));
                match e {
                    // A select's condition is never half; its values are as the select is.
                    Expr::Select { cond, if_true, if_false } => {
                        mark(*cond, false);
                        mark(*if_true, good_user);
                        mark(*if_false, good_user);
                    }
                    _ => e.for_each_value(&mut |x| mark(x, good_user)),
                }
            }
            Stmt::Store(p, v) => {
                let good =
                    p.path.is_empty() && p.root_local().is_some_and(|l| half_locals.contains(&l));
                p.for_each_value(&mut |x| mark(x, false));
                mark(*v, good);
            }
            Stmt::Return(Some(v)) => mark(*v, false),
            _ => s.for_each_value(&mut |x| mark(x, false)),
        });
        if let Some(c) = colour {
            ok.insert(c, half.contains(&c));
            // The returned construct reads it: that use is the colour sink.
        }
        let before = (half.len(), half_locals.len());
        half.retain(|v| ok.get(v).copied().unwrap_or(true) || Some(*v) == colour);
        // A load of a local is half if its value is; a local is half if all its loads are.
        half_locals.retain(|l| loads[l].iter().all(|v| half.contains(v)));
        // Loads of non-half locals aren't half.
        for (l, vs) in &loads {
            if !half_locals.contains(l) {
                for v in vs {
                    half.remove(v);
                }
            }
        }
        if (half.len(), half_locals.len()) == before {
            break;
        }
    }
    // Loads are where values arrive, not computation.
    half.retain(|v| !matches!(lets[v], Expr::Load(_)));
    half
}

/// The float values a fragment function computes that reach the colour of output 0 by pure
/// arithmetic and locals (each could have a half copy for that path, whatever else uses it).
pub fn reach_colour(m: &Module, f: &Function) -> HashSet<ValueId> {
    use wrela_ir::Scalar;
    let float = |t: TypeId| shape(&m.types, t).is_some_and(|(s, _)| s == Scalar::F32);
    let mut lets: HashMap<ValueId, Expr> = HashMap::new();
    let mut stores: HashMap<wrela_ir::LocalId, Vec<ValueId>> = HashMap::new();
    let mut ret = None;
    wrela_ir::visit::walk(&f.body, &mut |s| match s {
        Stmt::Let(v, e) => {
            lets.insert(*v, e.clone());
        }
        Stmt::Store(p, v) if p.path.is_empty() => {
            if let Some(l) = p.root_local() {
                stores.entry(l).or_default().push(*v);
            }
        }
        Stmt::Return(Some(v)) => ret = Some(*v),
        _ => {}
    });
    let mut out = HashSet::new();
    let mut stack: Vec<ValueId> = match ret.and_then(|r| lets.get(&r).cloned()) {
        Some(Expr::Construct(_, xs)) => xs.first().copied().into_iter().collect(),
        _ => vec![],
    };
    while let Some(v) = stack.pop() {
        if !out.insert(v) {
            continue;
        }
        let Some(e) = lets.get(&v) else { continue };
        match e {
            Expr::Load(p) if p.path.is_empty() && p.root_local().is_some() => {
                stack.extend(stores.get(&p.root_local().unwrap()).into_iter().flatten().copied());
            }
            Expr::Select { if_true, if_false, .. } => stack.extend([*if_true, *if_false]),
            e if e.only_computes() => e.for_each_value(&mut |x| {
                if float(f.value_ty(x)) {
                    stack.push(x);
                }
            }),
            _ => {}
        }
    }
    out.retain(|v| {
        lets.get(v).is_some_and(|e| e.only_computes() && !matches!(e, Expr::Const(_)))
            && float(f.value_ty(*v))
    });
    out
}
