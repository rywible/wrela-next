//! Debug builds' bounds checks on the GPU (language.md §11). WGSL doesn't trap: an index out of
//! range reads or writes some element in range, or a zero. In a debug build each dynamic index
//! in a pipeline's code is checked against its array's length first, and one out of range
//! records the pipeline's number (from 1) in a flag the host reads after each frame. The access
//! then goes on as WGSL defines it.
//!
//! Static proof erasure ([`crate::ranges::FlowEnv`]): before emitting a check, flow-sensitive
//! range analysis checks whether `0 <= i < len` is already proven by constants, bitmasks/modulo,
//! `for`-loop induction bounds, `if i < len` branches, or `if i >= len { return }` early-exit
//! guards. Proven-safe checks are erased at compile time; a module with no remaining checks needs
//! no flag resource at all.
//!
//! Vertex shaders can't write storage, so code a vertex entry point reaches isn't checked.

use crate::ranges::{FlowEnv, always_exits};
use crate::*;
use std::collections::HashSet;

/// Adds the checks to a flattened GPU module, with the flag (`array<atomic<u32>>`) at
/// `binding`, set to `code` (at least it: the flag keeps the largest). `None` if the module
/// has no dynamically unchecked index, so it needs no flag.
pub fn add_checks(m: &mut Module, binding: u32, code: u32) -> Option<ResourceId> {
    add_checks_stats(m, binding, code).0
}

/// As [`add_checks`], also returning `(emitted, erased)` check counts for diagnostics and tests.
pub fn add_checks_stats(
    m: &mut Module,
    binding: u32,
    code: u32,
) -> (Option<ResourceId>, usize, usize) {
    let skip = vertex_reach(m);
    let any = m.functions.iter().enumerate().any(|(i, f)| {
        let mut found = false;
        if !skip.contains(&FuncId(i as u32)) {
            visit::walk(&f.body, &mut |s| {
                s.for_each_place(&mut |p| found |= p.has_index());
                found |= matches!(s.expr(), Some(Expr::ExtractDyn(..)));
            });
        }
        found
    });
    if !any {
        return (None, 0, 0);
    }
    let u32_ty = m.types.u32();
    let had_flag = m.resources.iter().any(|r| r.name == crate::fuel::FLAG && r.binding == binding);
    let flag = crate::fuel::flag(m, binding);
    let mut emitted = 0usize;
    let mut erased = 0usize;
    for i in 0..m.functions.len() {
        if skip.contains(&FuncId(i as u32)) {
            continue;
        }
        let mut f = std::mem::take(&mut m.functions[i]);
        let body = std::mem::take(&mut f.body);
        let mut cx = Checks { m, f: &mut f, flag, code, u32_ty, emitted: 0, erased: 0 };
        let mut env = FlowEnv::new();
        let new_body = cx.block(body, &mut env);
        emitted += cx.emitted;
        erased += cx.erased;
        f.body = new_body;
        m.functions[i] = f;
    }
    if emitted == 0 {
        // Every dynamic index was statically proven in bounds: drop the unused flag if we added it.
        if !had_flag && flag.index() + 1 == m.resources.len() {
            m.resources.pop();
        }
        return (None, 0, erased);
    }
    (Some(flag), emitted, erased)
}

/// The functions a vertex entry point reaches.
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

struct Checks<'a> {
    m: &'a mut Module,
    f: &'a mut Function,
    flag: ResourceId,
    code: u32,
    u32_ty: TypeId,
    emitted: usize,
    erased: usize,
}

/// How long an indexed array is.
enum Len {
    Const(u32),
    /// A storage buffer's.
    Of(PlaceRoot),
}

impl Checks<'_> {
    fn block(&mut self, body: Block, env: &mut FlowEnv) -> Block {
        let mut out = Vec::with_capacity(body.len());
        for s in body {
            self.stmt(s, &mut out, env);
        }
        out
    }

    /// `s`, after a check of each index it takes that isn't statically proven in bounds.
    fn stmt(&mut self, s: Stmt, out: &mut Block, env: &mut FlowEnv) {
        match s {
            Stmt::If { cond, then, else_ } => {
                let then_exits = always_exits(&then);
                let else_exits = always_exits(&else_);
                let mut then_env = env.clone();
                then_env.narrow_cond(&self.m.types, self.f, cond, true);
                let then = self.block(then, &mut then_env);
                let mut else_env = env.clone();
                else_env.narrow_cond(&self.m.types, self.f, cond, false);
                let else_ = self.block(else_, &mut else_env);
                if then_exits && !else_exits {
                    *env = else_env;
                } else if else_exits && !then_exits {
                    *env = then_env;
                } else {
                    env.havoc_written(self.m, &then);
                    env.havoc_written(self.m, &else_);
                }
                out.push(Stmt::If { cond, then, else_ });
            }
            Stmt::Loop { body, continuing } => {
                let ind = env.detect_loop_induction(&self.m.types, self.f, &body, &continuing);
                env.havoc_written(self.m, &body);
                env.havoc_written(self.m, &continuing);
                let mut loop_env = env.clone();
                if let Some((local, range, _)) = ind {
                    loop_env.local_ints.insert(local, range);
                }
                let body = self.block(body, &mut loop_env);
                let mut cont_env = env.clone();
                let continuing = self.block(continuing, &mut cont_env);
                out.push(Stmt::Loop { body, continuing });
            }
            s => {
                let mut checks = Vec::new();
                s.for_each_place(&mut |p| self.place_checks(p, &mut checks));
                if let Some(Expr::ExtractDyn(x, i)) = s.expr()
                    && let Some(n) = self.m.types.index_len(self.f.value_ty(*x))
                {
                    checks.push((*i, Len::Const(n)));
                }
                for (i, len) in checks {
                    let proven = match &len {
                        Len::Const(n) => env.in_const_bounds(&self.m.types, self.f, i, *n),
                        Len::Of(root) => env.in_storage_bounds(&self.m.types, self.f, i, root),
                    };
                    if proven {
                        self.erased += 1;
                    } else {
                        self.emitted += 1;
                        self.check(out, i, len);
                    }
                }
                match &s {
                    Stmt::Let(v, e) => env.record_let(&self.m.types, self.f, *v, e),
                    Stmt::Store(p, v) => env.record_store(&self.m.types, self.f, p, *v),
                    Stmt::Eval(Expr::Call(g, args)) => {
                        if let Some(callee) = self.m.functions.get(g.index()) {
                            for (a, p) in args.iter().zip(&callee.params) {
                                if let (Arg::Place(Place { root: PlaceRoot::Local(l), .. }), true) =
                                    (a, p.mutable)
                                {
                                    env.local_ints.remove(l);
                                }
                            }
                        }
                    }
                    _ => {}
                }
                out.push(s);
            }
        }
    }

    /// The indices a place takes, each with its array's length.
    fn place_checks(&self, p: &Place, out: &mut Vec<(ValueId, Len)>) {
        let storage = matches!(&p.root, PlaceRoot::Resource(r)
            if matches!(self.m.resources[r.index()].kind,
                ResourceKind::StorageRead | ResourceKind::StorageReadWrite));
        if storage && let Some(Proj::Index(i)) = p.path.first() {
            // The flag's own index (a zero) isn't checked.
            if !matches!(p.root, PlaceRoot::Resource(r) if r == self.flag) {
                out.push((*i, Len::Of(p.root.clone())));
            }
        }
        let Some((mut t, path)) = self.m.place_start(self.f, p) else { return };
        for proj in path {
            if let Proj::Index(i) = proj
                && let Some(n) = self.m.types.index_len(t)
            {
                out.push((*i, Len::Const(n)));
            }
            let Some(next) = self.m.proj_ty(t, proj) else { return };
            t = next;
        }
    }

    /// `if i >= len { atomicMax(&flag[0], code) }`, with `i` as a `u32` (a negative `i32`
    /// is then out of range too).
    fn check(&mut self, out: &mut Block, i: ValueId, len: Len) {
        let u = self.u32_ty;
        let i = match self.m.types.get(self.f.value_ty(i)) {
            TypeDef::Scalar(Scalar::U32) => i,
            TypeDef::Scalar(Scalar::I32) => self.f.let_(out, u, Expr::Bitcast(i, Scalar::U32)),
            _ => self.f.let_(out, u, Expr::Convert(i, Scalar::U32)),
        };
        let len = match len {
            Len::Const(n) => self.f.let_(out, u, Expr::Const(Const::U32(n))),
            Len::Of(root) => self.f.let_(out, u, Expr::ArrayLength(Place::root(root))),
        };
        let b = self.m.types.bool();
        let bad = self.f.let_(out, b, Expr::Binary(BinOp::Ge, i, len));
        let mut then = Vec::new();
        let zero = self.f.let_(&mut then, u, Expr::Const(Const::U32(0)));
        let code = self.f.let_(&mut then, u, Expr::Const(Const::U32(self.code)));
        let at = Place { root: PlaceRoot::Resource(self.flag), path: vec![Proj::Index(zero)] };
        then.push(Stmt::Eval(Expr::Atomic(AtomicOp::Max, at, vec![code])));
        out.push(Stmt::If { cond: bad, then, else_: Vec::new() });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erases_proven_safe_storage_and_loop_bounds_checks() {
        let mut m = Module::default();
        let u32_ty = m.types.u32();
        let bool_ty = m.types.bool();
        let arr_ty = m.types.intern(TypeDef::Array(u32_ty, 8));
        let buf = m.add_resource(Resource {
            name: "buf".into(),
            binding: 0,
            kind: ResourceKind::StorageReadWrite,
            ty: u32_ty,
        });
        let mut f = Function::new("k", vec![Param::value("id", u32_ty)], None);
        let arr = f.new_local("arr", arr_ty);
        let idx = f.new_local("i", u32_ty);
        let mut body = Vec::new();
        let id = f.let_(&mut body, u32_ty, Expr::Param(0));
        // 1. Storage guard: `if id >= len(buf) { return }` -> `buf[id]` is proven safe!
        let len =
            f.let_(&mut body, u32_ty, Expr::ArrayLength(Place::root(PlaceRoot::Resource(buf))));
        let oob = f.let_(&mut body, bool_ty, Expr::Binary(BinOp::Ge, id, len));
        body.push(Stmt::If { cond: oob, then: vec![Stmt::Return(None)], else_: Vec::new() });
        let zero = f.let_(&mut body, u32_ty, Expr::Const(Const::U32(0)));
        let buf_at = Place { root: PlaceRoot::Resource(buf), path: vec![Proj::Index(id)] };
        body.push(Stmt::Store(buf_at, zero));
        // 2. Canonical `for i in 0..8 { arr[i] = 0 }` -> `arr[i]` is proven safe!
        body.push(Stmt::Store(Place::local(idx), zero));
        let mut lbody = Vec::new();
        let iv = f.let_(&mut lbody, u32_ty, Expr::Load(Place::local(idx)));
        let eight = f.let_(&mut lbody, u32_ty, Expr::Const(Const::U32(8)));
        let done = f.let_(&mut lbody, bool_ty, Expr::Binary(BinOp::Ge, iv, eight));
        lbody.push(Stmt::If { cond: done, then: vec![Stmt::Break], else_: Vec::new() });
        let arr_at = Place { root: PlaceRoot::Local(arr), path: vec![Proj::Index(iv)] };
        lbody.push(Stmt::Store(arr_at, zero));
        let mut cont = Vec::new();
        let iv2 = f.let_(&mut cont, u32_ty, Expr::Load(Place::local(idx)));
        let one = f.let_(&mut cont, u32_ty, Expr::Const(Const::U32(1)));
        let next = f.let_(&mut cont, u32_ty, Expr::Binary(BinOp::Add, iv2, one));
        cont.push(Stmt::Store(Place::local(idx), next));
        body.push(Stmt::Loop { body: lbody, continuing: cont });
        f.body = body;
        let fid = m.add_function(f);
        m.entry_points.push(EntryPoint {
            name: "k".into(),
            stage: Stage::Compute { workgroup_size: [64, 1, 1] },
            function: fid,
            inputs: Vec::new(),
        });
        let (flag, emitted, erased) = add_checks_stats(&mut m, 7, 1);
        assert!(flag.is_none(), "all bounds checks should be erased so no flag is needed");
        assert_eq!(emitted, 0);
        assert_eq!(erased, 2);
        verify(&m).expect("valid IR");
    }

    #[test]
    fn keeps_unproven_bounds_check() {
        let mut m = Module::default();
        let u32_ty = m.types.u32();
        let buf = m.add_resource(Resource {
            name: "buf".into(),
            binding: 0,
            kind: ResourceKind::StorageReadWrite,
            ty: u32_ty,
        });
        let mut f = Function::new("k", vec![Param::value("id", u32_ty)], None);
        let mut body = Vec::new();
        let id = f.let_(&mut body, u32_ty, Expr::Param(0));
        let zero = f.let_(&mut body, u32_ty, Expr::Const(Const::U32(0)));
        let buf_at = Place { root: PlaceRoot::Resource(buf), path: vec![Proj::Index(id)] };
        body.push(Stmt::Store(buf_at, zero));
        f.body = body;
        let fid = m.add_function(f);
        m.entry_points.push(EntryPoint {
            name: "k".into(),
            stage: Stage::Compute { workgroup_size: [64, 1, 1] },
            function: fid,
            inputs: Vec::new(),
        });
        let (flag, emitted, erased) = add_checks_stats(&mut m, 7, 1);
        assert!(flag.is_some());
        assert_eq!(emitted, 1);
        assert_eq!(erased, 0);
        verify(&m).expect("valid IR");
    }
}
