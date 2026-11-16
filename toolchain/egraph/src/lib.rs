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
             uniform work {:.0} -> {:.0}; {} e-nodes; {:.1} ms; applied: {}",
            self.stretches,
            self.rewritten,
            self.limited,
            self.lets_before,
            self.lets_after,
            self.varying_before,
            self.varying_after,
            self.uniform_before,
            self.uniform_after,
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
        };
        cx.function(&mut f);
        m.functions[i] = f;
    }
    m.types = types;
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
        let mut uses: HashMap<ValueId, u32> = HashMap::new();
        wrela_ir::visit::walk(&f.body, &mut |s| {
            if let Stmt::Let(v, Expr::Const(c)) = s
                && let Some(l) = Lit::from_const(c)
            {
                consts.insert(*v, l);
            }
            s.for_each_value(&mut |v| *uses.entry(v).or_default() += 1);
        });
        let mut st = FnState { consts, uses, rename: HashMap::new() };
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
        if after.cost + 1e-9 >= before.cost {
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
        let mut emitted: HashMap<Id, ValueId> = HashMap::new();
        let n0 = out.len();
        for (v, &c) in roots.iter().zip(&roots_now) {
            let nv = emit(&eg, &best, &exprs, f, out, &mut emitted, c);
            st.rename.insert(*v, nv);
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

/// Writes the class `c` (and what it needs) as lets, once each; its value.
fn emit(
    eg: &Graph,
    best: &HashMap<Id, Node>,
    exprs: &HashMap<u32, Expr>,
    f: &mut Function,
    out: &mut Vec<Stmt>,
    emitted: &mut HashMap<Id, ValueId>,
    c: Id,
) -> ValueId {
    let c = eg.find(c);
    if let Some(&v) = emitted.get(&c) {
        return v;
    }
    let n = best[&c].clone();
    let kids: Vec<ValueId> =
        n.children().iter().map(|&k| emit(eg, best, exprs, f, out, emitted, k)).collect();
    let e = match &n {
        Node::Leaf(x) => {
            emitted.insert(c, ValueId(*x));
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
            let mut e = exprs[x].clone();
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
    emitted.insert(c, v);
    v
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
