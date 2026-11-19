//! What a function's result depends on (language.md §14, `std::derive::reads_only_input`): which
//! of its parameters, and whether anything outside them (a resource, a builtin input, the host,
//! another invocation). A value depends on what it's computed from, and on the condition of every
//! branch it's computed under (a value a branch on a parameter chooses depends on that parameter),
//! across the calls it makes.
//!
//! The answer is an over-approximation: a result said not to depend on a parameter never does. A
//! local depends on everything ever stored to it (the order of stores isn't followed), and every
//! value computed in a loop depends on the conditions that can leave it.

use crate::*;
use std::collections::HashMap;

/// What a result depends on: a bit a parameter (the first 63), and `OUTSIDE`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Deps(u64);

/// Anything but the function's parameters: a resource, a builtin input, the host, a derivative
/// (its neighbours' values), a subgroup or an atomic.
const OUTSIDE: u64 = 1 << 63;

impl Deps {
    fn param(i: u32) -> Deps {
        // Past the 63rd parameter, every one counts as all of them: never too few.
        if i < 63 { Deps(1 << i) } else { Deps(!0) }
    }

    /// Whether it depends on parameter `i`.
    pub fn on_param(self, i: u32) -> bool {
        self.0 & Deps::param(i).0 & !OUTSIDE != 0
    }

    /// Whether it depends on anything outside the parameters.
    pub fn outside(self) -> bool {
        self.0 & OUTSIDE != 0
    }

    fn or(self, o: Deps) -> Deps {
        Deps(self.0 | o.0)
    }
}

/// What the result of function `f` of module `m` depends on.
pub fn depends(m: &Module, f: FuncId) -> Deps {
    let mut memo = HashMap::new();
    summary(m, f, &mut memo, 0)
}

fn summary(m: &Module, f: FuncId, memo: &mut HashMap<u32, Deps>, depth: u32) -> Deps {
    if let Some(d) = memo.get(&f.0) {
        return *d;
    }
    // GPU code doesn't recurse; a cycle (or a chain past any real one) answers everything.
    if depth > 256 {
        return Deps(!0);
    }
    memo.insert(f.0, Deps(!0));
    let func = &m.functions[f.index()];
    let mut w = Walk {
        m,
        memo,
        depth,
        vals: HashMap::new(),
        locals: HashMap::new(),
        result: Deps::default(),
        changed: true,
    };
    while w.changed {
        w.changed = false;
        w.block(&func.body, Deps::default());
    }
    let r = w.result;
    memo.insert(f.0, r);
    r
}

struct Walk<'a, 'm> {
    m: &'a Module,
    memo: &'m mut HashMap<u32, Deps>,
    depth: u32,
    vals: HashMap<ValueId, Deps>,
    locals: HashMap<LocalId, Deps>,
    result: Deps,
    changed: bool,
}

impl Walk<'_, '_> {
    fn val(&self, v: ValueId) -> Deps {
        self.vals.get(&v).copied().unwrap_or_default()
    }

    fn set_val(&mut self, v: ValueId, d: Deps) {
        let e = self.vals.entry(v).or_default();
        let n = e.or(d);
        if n != *e {
            *e = n;
            self.changed = true;
        }
    }

    fn set_local(&mut self, l: LocalId, d: Deps) {
        let e = self.locals.entry(l).or_default();
        let n = e.or(d);
        if n != *e {
            *e = n;
            self.changed = true;
        }
    }

    /// What reading `p` depends on: its root and its indices.
    fn place(&self, p: &Place) -> Deps {
        let mut d = match &p.root {
            PlaceRoot::Local(l) => self.locals.get(l).copied().unwrap_or_default(),
            PlaceRoot::Param(i) => Deps::param(*i),
            PlaceRoot::Data(_) => Deps::default(),
            PlaceRoot::Resource(_) | PlaceRoot::Ptr(_) => Deps(OUTSIDE),
        };
        for x in &p.path {
            if let Proj::Index(i) = x {
                d = d.or(self.val(*i));
            }
        }
        d
    }

    fn expr(&mut self, e: &Expr, ctx: Deps) -> Deps {
        let mut d = ctx;
        match e {
            Expr::Const(_) | Expr::Zero(_) => {}
            Expr::Param(i) => d = d.or(Deps::param(*i)),
            Expr::Load(p) => d = d.or(self.place(p)),
            Expr::Call(g, args) => {
                let s = summary(self.m, *g, self.memo, self.depth + 1);
                if s.outside() {
                    d = d.or(Deps(OUTSIDE));
                }
                for (i, a) in args.iter().enumerate() {
                    let ad = match a {
                        Arg::Value(v) => self.val(*v),
                        Arg::Place(p) => self.place(p),
                    };
                    if s.on_param(i as u32) {
                        d = d.or(ad);
                    }
                }
                // A place lent to it may be written there, from anything it was given.
                let all = args.iter().fold(d, |acc, a| match a {
                    Arg::Value(v) => acc.or(self.val(*v)),
                    Arg::Place(p) => acc.or(self.place(p)),
                });
                for a in args {
                    if let Arg::Place(p) = a
                        && let PlaceRoot::Local(l) = p.root
                    {
                        self.set_local(l, all);
                    }
                }
            }
            Expr::Builtin(b, _) if matches!(b, Builtin::Dpdx | Builtin::Dpdy | Builtin::Fwidth) => {
                d = d.or(Deps(OUTSIDE));
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
            | Expr::ArrayLength(_) => d = d.or(Deps(OUTSIDE)),
            _ => {}
        }
        let mut vs = d;
        e.for_each_value(&mut |v| vs = vs.or(self.val(v)));
        vs
    }

    fn block(&mut self, b: &[Stmt], ctx: Deps) {
        for s in b {
            match s {
                Stmt::Let(v, e) => {
                    let d = self.expr(e, ctx);
                    self.set_val(*v, d);
                }
                Stmt::Eval(e) => {
                    self.expr(e, ctx);
                }
                Stmt::Store(p, v) => {
                    if let PlaceRoot::Local(l) = p.root {
                        let d = ctx.or(self.val(*v)).or(self.place(p));
                        self.set_local(l, d);
                    }
                }
                Stmt::If { cond, then, else_ } => {
                    let c = ctx.or(self.val(*cond));
                    self.block(then, c);
                    self.block(else_, c);
                }
                Stmt::Loop { body, continuing } => {
                    // Every pass, and what follows, may depend on any condition in the loop.
                    let mut c = ctx;
                    visit::walk(body, &mut |s| {
                        if let Stmt::If { cond, .. } = s {
                            c = c.or(self.val(*cond));
                        }
                    });
                    // (What follows the loop reads what it stored, which carries `c`.)
                    self.block(body, c);
                    self.block(continuing, c);
                }
                Stmt::Return(v) => {
                    let d = ctx.or(v.map_or(Deps::default(), |v| self.val(v)));
                    let n = self.result.or(d);
                    if n != self.result {
                        self.result = n;
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

    fn module(body: Vec<Stmt>, params: usize) -> (Module, FuncId) {
        let mut m = Module::default();
        let f32t = m.types.f32();
        let ps = (0..params).map(|i| Param::value(&format!("p{i}"), f32t)).collect();
        let mut f = Function::new("f", ps, Some(f32t));
        f.body = body;
        let id = m.add_function(f);
        (m, id)
    }

    #[test]
    fn a_branch_on_a_parameter_makes_the_result_depend_on_it() {
        let (a, b, c, r) = (ValueId(0), ValueId(1), ValueId(2), ValueId(3));
        let one = ValueId(4);
        let body = vec![
            Stmt::Let(a, Expr::Param(0)),
            Stmt::Let(b, Expr::Param(1)),
            Stmt::Let(one, Expr::Const(Const::F32(1.0))),
            Stmt::Let(c, Expr::Binary(BinOp::Lt, b, one)),
            Stmt::If { cond: c, then: vec![Stmt::Return(Some(a))], else_: vec![] },
            Stmt::Let(r, Expr::Binary(BinOp::Add, a, one)),
            Stmt::Return(Some(r)),
        ];
        let (m, f) = module(body, 3);
        let d = depends(&m, f);
        assert!(d.on_param(0) && d.on_param(1) && !d.on_param(2) && !d.outside(), "{d:?}");
    }
}
