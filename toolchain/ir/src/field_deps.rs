//! What a function's result depends on, part by part of its parameters (language.md §14,
//! `std::derive::kept_deps`): [`crate::depends`] says whether a result depends on a parameter at
//! all; this says on which of its fields, components and elements, followed through the fields a
//! value is built of and taken apart into, the locals it is kept in, and the calls it is passed
//! to (each callee's own answer, in terms of its parameters' parts, read at the parts it was
//! given).
//!
//! A value is a tree: a part of a parameter (`Sym`: all of parameter `p` under `path`), a value
//! built of parts (`Node`, by field, component or element), or a value computed from others
//! (`Leaf`: everything it was computed from, flattened). What a computed value depends on is
//! everything its operands depend on, and the condition of every branch it is computed under. The
//! answer over-approximates: a part said not to matter never does. Where the tree can't follow a
//! value (a dynamic index, an enum, a pointer), it takes all of what the value holds.

use crate::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// A path into a parameter: struct fields, vector components and array elements (by constant
/// index), as `Expr::Extract` numbers them.
pub type Path = Vec<u32>;

/// What a value depends on: parts of parameters (each all of parameter `.0` under path `.1`),
/// and whether anything outside them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DepSet {
    pub parts: BTreeSet<(u32, Path)>,
    pub outside: bool,
}

impl DepSet {
    fn outside() -> DepSet {
        DepSet { parts: BTreeSet::new(), outside: true }
    }

    fn join(&mut self, o: &DepSet) {
        self.outside |= o.outside;
        for p in &o.parts {
            self.parts.insert(p.clone());
        }
    }

    fn joined(mut self, o: &DepSet) -> DepSet {
        self.join(o);
        self
    }

    /// Whether it depends on any part of parameter `i`.
    pub fn on_param(&self, i: u32) -> bool {
        self.parts.iter().any(|(p, _)| *p == i)
    }

    /// The paths of parameter `i` it depends on, none a prefix of another.
    pub fn paths_of(&self, i: u32) -> Vec<Path> {
        let mut out: Vec<Path> = Vec::new();
        for (p, path) in &self.parts {
            if *p == i {
                out.push(path.clone());
            }
        }
        out.sort();
        let mut kept: Vec<Path> = Vec::new();
        for path in out {
            if !kept.iter().any(|k| path.starts_with(k)) {
                kept.push(path);
            }
        }
        kept
    }
}

/// A value's dependence, part by part.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Tree {
    /// Computed: depends on all of this.
    Leaf(DepSet),
    /// All of parameter `.0` under path `.1`, and `.2` besides (the branches it was chosen under).
    Sym(u32, Path, DepSet),
    /// Built of parts, by index (a missing part depends on nothing).
    Node(BTreeMap<u32, Tree>, DepSet),
}

/// How deep a value's parts are followed: past it, a value is flattened (no type nests deeper).
const DEEPEST: usize = 12;

impl Tree {
    fn empty() -> Tree {
        Tree::Leaf(DepSet::default())
    }

    fn flat(&self) -> DepSet {
        match self {
            Tree::Leaf(d) => d.clone(),
            Tree::Sym(p, path, x) => {
                let mut d = x.clone();
                d.parts.insert((*p, path.clone()));
                d
            }
            Tree::Node(kids, x) => {
                let mut d = x.clone();
                for k in kids.values() {
                    d.join(&k.flat());
                }
                d
            }
        }
    }

    fn with_ctx(self, ctx: &DepSet) -> Tree {
        if ctx.parts.is_empty() && !ctx.outside {
            return self;
        }
        match self {
            Tree::Leaf(d) => Tree::Leaf(d.joined(ctx)),
            Tree::Sym(p, path, x) => Tree::Sym(p, path, x.joined(ctx)),
            Tree::Node(kids, x) => Tree::Node(kids, x.joined(ctx)),
        }
    }

    /// Part `k`.
    fn part(&self, k: u32) -> Tree {
        match self {
            Tree::Leaf(d) => Tree::Leaf(d.clone()),
            Tree::Sym(p, path, x) => {
                if path.len() >= DEEPEST {
                    return Tree::Sym(*p, path.clone(), x.clone());
                }
                let mut q = path.clone();
                q.push(k);
                Tree::Sym(*p, q, x.clone())
            }
            Tree::Node(kids, x) => kids.get(&k).cloned().unwrap_or_else(Tree::empty).with_ctx(x),
        }
    }

    /// Any part (a dynamic index): a part of a parameter's array stays the whole array.
    fn any_part(&self) -> Tree {
        match self {
            Tree::Leaf(_) | Tree::Sym(..) => self.clone(),
            Tree::Node(..) => Tree::Leaf(self.flat()),
        }
    }

    fn at(&self, path: &[u32]) -> Tree {
        let mut t = self.clone();
        for k in path {
            t = t.part(*k);
        }
        t
    }

    /// Whether it depends on nothing at all.
    fn is_nothing(&self) -> bool {
        matches!(self, Tree::Leaf(d) if d.parts.is_empty() && !d.outside)
    }

    fn join(&self, o: &Tree) -> Tree {
        // What depends on nothing adds nothing (a local before its first store).
        if self.is_nothing() {
            return o.clone();
        }
        if o.is_nothing() {
            return self.clone();
        }
        match (self, o) {
            (Tree::Leaf(a), Tree::Leaf(b)) if a == b => self.clone(),
            (Tree::Sym(p, a, x), Tree::Sym(q, b, y)) if p == q && a == b => {
                Tree::Sym(*p, a.clone(), x.clone().joined(y))
            }
            (Tree::Node(a, x), Tree::Node(b, y)) => {
                let mut kids = a.clone();
                for (k, t) in b {
                    let n = match kids.get(k) {
                        Some(s) => s.join(t),
                        None => t.clone(),
                    };
                    kids.insert(*k, n);
                }
                Tree::Node(kids, x.clone().joined(y))
            }
            _ => Tree::Leaf(self.flat().joined(&o.flat())),
        }
    }

    /// The tree with `v` joined in at `path`.
    fn joined_at(&self, path: &[Option<u32>], v: &Tree) -> Tree {
        match path.first() {
            None => self.join(v),
            // A dynamic index: every element may be it.
            Some(None) => Tree::Leaf(self.flat().joined(&v.flat())),
            Some(Some(k)) => match self {
                Tree::Node(kids, x) => {
                    let mut kids = kids.clone();
                    let old = kids.get(k).cloned().unwrap_or_else(Tree::empty);
                    kids.insert(*k, old.joined_at(&path[1..], v));
                    Tree::Node(kids, x.clone())
                }
                // A local not yet written: built of what is stored in it.
                Tree::Leaf(d) if d.parts.is_empty() && !d.outside => {
                    let mut kids = BTreeMap::new();
                    kids.insert(*k, Tree::empty().joined_at(&path[1..], v));
                    Tree::Node(kids, DepSet::default())
                }
                _ => Tree::Leaf(self.flat().joined(&v.flat())),
            },
        }
    }

    /// A callee's result (in its parameters' parts) read at the trees of what it was given.
    fn given(&self, args: &[Tree]) -> Tree {
        match self {
            Tree::Leaf(d) => Tree::Leaf(given_set(d, args)),
            Tree::Sym(p, path, x) => {
                let a = args.get(*p as usize).map(|a| a.at(path)).unwrap_or(Tree::Leaf(DepSet::outside()));
                a.with_ctx(&given_set(x, args))
            }
            Tree::Node(kids, x) => {
                let kids = kids.iter().map(|(k, t)| (*k, t.given(args))).collect();
                Tree::Node(kids, given_set(x, args))
            }
        }
    }
}

fn given_set(d: &DepSet, args: &[Tree]) -> DepSet {
    let mut out = DepSet { parts: BTreeSet::new(), outside: d.outside };
    for (p, path) in &d.parts {
        match args.get(*p as usize) {
            Some(a) => out.join(&a.at(path).flat()),
            None => out.outside = true,
        }
    }
    out
}

/// What the result of function `f` of module `m` depends on, part by part of its parameters
/// (`DepSet::paths_of`).
pub fn depends_fields(m: &Module, f: FuncId) -> DepSet {
    let mut memo = HashMap::new();
    summary(m, f, &mut memo, 0).flat()
}

fn everything(m: &Module, f: FuncId) -> Tree {
    let n = m.functions[f.index()].params.len() as u32;
    let mut d = DepSet::outside();
    for i in 0..n {
        d.parts.insert((i, Vec::new()));
    }
    Tree::Leaf(d)
}

fn summary(m: &Module, f: FuncId, memo: &mut HashMap<u32, Tree>, depth: u32) -> Tree {
    if let Some(t) = memo.get(&f.0) {
        return t.clone();
    }
    // GPU code doesn't recurse; a cycle (or a chain past any real one) answers everything.
    if depth > 256 {
        return everything(m, f);
    }
    memo.insert(f.0, everything(m, f));
    let func = &m.functions[f.index()];
    let mut w = Walk {
        m,
        memo,
        depth,
        vals: HashMap::new(),
        locals: HashMap::new(),
        result: None,
        changed: true,
        passes: 0,
    };
    while w.changed {
        w.changed = false;
        w.passes += 1;
        if w.passes > 64 {
            let all = everything(m, f);
            w.memo.insert(f.0, all.clone());
            return all;
        }
        w.block(&func.body, &DepSet::default());
    }
    let r = w.result.unwrap_or_else(Tree::empty);
    w.memo.insert(f.0, r.clone());
    r
}

struct Walk<'a, 'm> {
    m: &'a Module,
    memo: &'m mut HashMap<u32, Tree>,
    depth: u32,
    vals: HashMap<ValueId, Tree>,
    locals: HashMap<LocalId, Tree>,
    result: Option<Tree>,
    changed: bool,
    passes: u32,
}

impl Walk<'_, '_> {
    fn val(&self, v: ValueId) -> Tree {
        self.vals.get(&v).cloned().unwrap_or_else(Tree::empty)
    }

    fn set_val(&mut self, v: ValueId, t: Tree) {
        let n = match self.vals.get(&v) {
            Some(old) => old.join(&t),
            None => t,
        };
        if self.vals.get(&v) != Some(&n) {
            self.vals.insert(v, n);
            self.changed = true;
        }
    }

    fn store_local(&mut self, l: LocalId, path: &[Option<u32>], t: &Tree) {
        let old = self.locals.get(&l).cloned().unwrap_or_else(Tree::empty);
        let n = old.joined_at(path, t);
        if self.locals.get(&l) != Some(&n) {
            self.locals.insert(l, n);
            self.changed = true;
        }
    }

    /// A place's path as parts (`None` for a dynamic index), and what its indices depend on.
    fn path(&self, p: &Place) -> (Vec<Option<u32>>, DepSet) {
        let mut out = Vec::new();
        let mut d = DepSet::default();
        for x in &p.path {
            match x {
                Proj::Field(k) => out.push(Some(*k)),
                Proj::Comp(c) => out.push(Some(*c as u32)),
                Proj::Index(i) => {
                    d.join(&self.val(*i).flat());
                    out.push(None);
                }
            }
        }
        (out, d)
    }

    fn place(&self, p: &Place) -> Tree {
        let root = match &p.root {
            PlaceRoot::Local(l) => self.locals.get(l).cloned().unwrap_or_else(Tree::empty),
            PlaceRoot::Param(i) => Tree::Sym(*i, Vec::new(), DepSet::default()),
            PlaceRoot::Data(_) => Tree::empty(),
            PlaceRoot::Resource(_) | PlaceRoot::Ptr(_) => Tree::Leaf(DepSet::outside()),
        };
        let (path, idx) = self.path(p);
        let mut t = root;
        for k in path {
            t = match k {
                Some(k) => t.part(k),
                None => t.any_part(),
            };
        }
        t.with_ctx(&idx)
    }

    fn operands(&self, e: &Expr) -> DepSet {
        let mut d = DepSet::default();
        e.for_each_value(&mut |v| d.join(&self.val(v).flat()));
        d
    }

    fn expr(&mut self, e: &Expr, ctx: &DepSet) -> Tree {
        let t = match e {
            Expr::Const(_) | Expr::Zero(_) => Tree::empty(),
            Expr::Param(i) => Tree::Sym(*i, Vec::new(), DepSet::default()),
            Expr::Load(p) => self.place(p),
            Expr::Extract(v, k) => self.val(*v).part(*k),
            Expr::ExtractDyn(v, i) => self.val(*v).any_part().with_ctx(&self.val(*i).flat()),
            Expr::Swizzle(v, comps) => {
                let t = self.val(*v);
                let kids = comps.iter().enumerate().map(|(j, c)| (j as u32, t.part(*c as u32))).collect();
                Tree::Node(kids, DepSet::default())
            }
            Expr::Construct(_, vs) => {
                let kids = vs.iter().enumerate().map(|(j, v)| (j as u32, self.val(*v))).collect();
                Tree::Node(kids, DepSet::default())
            }
            Expr::Select { cond, if_true, if_false } => {
                self.val(*if_true).join(&self.val(*if_false)).with_ctx(&self.val(*cond).flat())
            }
            Expr::Call(g, args) => {
                let s = summary(self.m, *g, self.memo, self.depth + 1);
                let given: Vec<Tree> = args
                    .iter()
                    .map(|a| match a {
                        Arg::Value(v) => self.val(*v),
                        Arg::Place(p) => self.place(p),
                    })
                    .collect();
                let r = s.given(&given);
                // A place lent to it may be written there, from anything it was given.
                let mut all = r.flat();
                for g in &given {
                    all.join(&g.flat());
                }
                let all = Tree::Leaf(all.joined(ctx));
                for a in args {
                    if let Arg::Place(p) = a
                        && let PlaceRoot::Local(l) = p.root
                    {
                        let (path, _) = self.path(p);
                        self.store_local(l, &path, &all);
                    }
                }
                r
            }
            Expr::Builtin(b, _) if matches!(b, Builtin::Dpdx | Builtin::Dpdy | Builtin::Fwidth) => {
                Tree::Leaf(self.operands(e).joined(&DepSet::outside()))
            }
            Expr::EntryInput(_)
            | Expr::Host(..)
            | Expr::Mem(..)
            | Expr::Run(_)
            | Expr::Addr(_)
            | Expr::Texture(..)
            | Expr::TextureStore(..)
            | Expr::Atomic(..)
            | Expr::Subgroup(..)
            | Expr::Barrier
            | Expr::Discard
            | Expr::ArrayLength(_) => Tree::Leaf(self.operands(e).joined(&DepSet::outside())),
            _ => Tree::Leaf(self.operands(e)),
        };
        t.with_ctx(ctx)
    }

    fn block(&mut self, b: &[Stmt], ctx: &DepSet) {
        for s in b {
            match s {
                Stmt::Let(v, e) => {
                    let t = self.expr(e, ctx);
                    self.set_val(*v, t);
                }
                Stmt::Eval(e) => {
                    self.expr(e, ctx);
                }
                Stmt::Store(p, v) => {
                    if let PlaceRoot::Local(l) = p.root {
                        let (path, idx) = self.path(p);
                        let t = self.val(*v).with_ctx(&ctx.clone().joined(&idx));
                        self.store_local(l, &path, &t);
                    }
                }
                Stmt::If { cond, then, else_ } => {
                    let c = ctx.clone().joined(&self.val(*cond).flat());
                    self.block(then, &c);
                    self.block(else_, &c);
                }
                Stmt::Loop { body, continuing } => {
                    // Every pass, and what follows, may depend on any condition in the loop.
                    let mut c = ctx.clone();
                    visit::walk(body, &mut |s| {
                        if let Stmt::If { cond, .. } = s {
                            c.join(&self.val(*cond).flat());
                        }
                    });
                    self.block(body, &c);
                    self.block(continuing, &c);
                }
                Stmt::Return(v) => {
                    let t = v.map_or_else(Tree::empty, |v| self.val(v)).with_ctx(ctx);
                    let n = match &self.result {
                        Some(r) => r.join(&t),
                        None => t,
                    };
                    if self.result.as_ref() != Some(&n) {
                        self.result = Some(n);
                        self.changed = true;
                    }
                }
                Stmt::Break | Stmt::Continue | Stmt::Trap | Stmt::At(_) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_result_depends_on_the_fields_it_reads_and_not_the_rest() {
        let mut m = Module::default();
        let f32t = m.types.f32();
        let v4 = m.types.intern(TypeDef::Vector(Scalar::F32, 4));
        let s = m.types.intern(TypeDef::Struct {
            name: "S".into(),
            fields: vec![("a".into(), f32t), ("glow".into(), v4)],
        });
        // f(s) = s.a + s.glow.w; the glow's colour (x, y, z) is computed and dropped.
        let (p, a, g, w, x, r) = (ValueId(0), ValueId(1), ValueId(2), ValueId(3), ValueId(4), ValueId(5));
        let body = vec![
            Stmt::Let(p, Expr::Param(0)),
            Stmt::Let(a, Expr::Extract(p, 0)),
            Stmt::Let(g, Expr::Extract(p, 1)),
            Stmt::Let(w, Expr::Extract(g, 3)),
            Stmt::Let(x, Expr::Extract(g, 0)),
            Stmt::Let(r, Expr::Binary(BinOp::Add, a, w)),
            Stmt::Return(Some(r)),
        ];
        let mut func = Function::new("f", vec![Param::value("s", s)], Some(f32t));
        func.body = body;
        let id = m.add_function(func);
        let d = depends_fields(&m, id);
        assert_eq!(d.paths_of(0), vec![vec![0], vec![1, 3]], "{d:?}");
        assert!(!d.outside);
    }

    #[test]
    fn a_parameter_kept_in_a_local_is_followed_part_by_part() {
        let mut m = Module::default();
        let f32t = m.types.f32();
        let v4 = m.types.intern(TypeDef::Vector(Scalar::F32, 4));
        let s = m.types.intern(TypeDef::Struct {
            name: "S".into(),
            fields: vec![("a".into(), f32t), ("glow".into(), v4)],
        });
        let (p, x) = (ValueId(0), ValueId(1));
        let mut func = Function::new("f", vec![Param::value("s", s)], Some(f32t));
        let l = func.new_local("s", s);
        func.body = vec![
            Stmt::Let(p, Expr::Param(0)),
            Stmt::Store(Place::local(l), p),
            Stmt::Let(x, Expr::Load(Place::local(l).with(Proj::Field(1)).with(Proj::Comp(3)))),
            Stmt::Return(Some(x)),
        ];
        let id = m.add_function(func);
        let d = depends_fields(&m, id);
        assert_eq!(d.paths_of(0), vec![vec![1, 3]], "{d:?}");
    }

    #[test]
    fn a_callee_is_read_at_the_parts_it_was_given() {
        let mut m = Module::default();
        let f32t = m.types.f32();
        let v4 = m.types.intern(TypeDef::Vector(Scalar::F32, 4));
        // g(v) = v.y
        let (gp, gy) = (ValueId(0), ValueId(1));
        let mut gf = Function::new("g", vec![Param::value("v", v4)], Some(f32t));
        gf.body = vec![Stmt::Let(gp, Expr::Param(0)), Stmt::Let(gy, Expr::Extract(gp, 1)), Stmt::Return(Some(gy))];
        let g = m.add_function(gf);
        // f(a: vec4, b: vec4) = g(b)
        let (fb, fr) = (ValueId(0), ValueId(1));
        let mut ff = Function::new("f", vec![Param::value("a", v4), Param::value("b", v4)], Some(f32t));
        ff.body = vec![
            Stmt::Let(fb, Expr::Param(1)),
            Stmt::Let(fr, Expr::Call(g, vec![Arg::Value(fb)])),
            Stmt::Return(Some(fr)),
        ];
        let f = m.add_function(ff);
        let d = depends_fields(&m, f);
        assert!(!d.on_param(0), "{d:?}");
        assert_eq!(d.paths_of(1), vec![vec![1]], "{d:?}");
    }
}
