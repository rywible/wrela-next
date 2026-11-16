//! WGSL's uniformity rule for derivatives and barriers (WGSL §15.2), checked on a fragment
//! shader's or a kernel's module before it's flattened.
//!
//! `dpdx`, `dpdy` and `fwidth` compare a pixel with its neighbours, and a barrier waits for
//! every invocation of the workgroup, so WGSL requires every invocation to reach them together:
//! in uniform control flow. Control flow stops being uniform inside a branch on a value that
//! may differ between invocations (a varying, the fragment's position, a value computed from
//! one, a variable assigned in such a branch). Branches that only fall through meet again after
//! the `if`, and a loop's iterations meet again after the loop; but after a branch that may
//! `return`, `break` or `continue`, the flow stays non-uniform until the function (or the loop)
//! ends.
//!
//! A subgroup operation needs less: only its subgroup's invocations must reach it together. So
//! each value and the flow have a [`Level`]: the same in the whole workgroup, the same within
//! each subgroup, or different between invocations. What a subgroup's reduction gives
//! (`subgroup_any`, `subgroup_min`, `subgroup_broadcast_first`, a shuffle from one place) is the
//! same within the subgroup, so a branch on it, even one that leaves a loop, keeps a subgroup's
//! invocations together: a loop runs while any of a subgroup's invocations has work. The WGSL
//! back end turns WGSL's own, workgroup-wide check of subgroup operations off
//! (`subgroup_uniformity`), as this one stands for it.
//!
//! The analysis follows calls into their callees with the caller's flow, as flattening will
//! inline them (`opt`), so it sees the code WGSL will see.

use crate::*;
use std::collections::{HashMap, HashSet};
use wrela_diag::Span;

/// A collective operation (WGSL's term): the invocations that run it must all reach it
/// together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Collective {
    Dpdx,
    Dpdy,
    Fwidth,
    /// `textureSample`, `textureSampleBias` and `textureSampleCompare`: they take derivatives
    /// to pick the detail.
    Sample,
    SampleBias,
    SampleCompare,
    /// `workgroupBarrier`: for all the invocations of a workgroup.
    Barrier,
    /// A subgroup operation: for all the invocations of a subgroup.
    Subgroup(crate::SubgroupOp),
}

impl Collective {
    /// How uniform the flow that reaches it must be: a subgroup operation's subgroup's, the
    /// rest the workgroup's.
    fn needs(self) -> Level {
        match self {
            Collective::Subgroup(_) => SUBGROUP,
            _ => WORKGROUP,
        }
    }
}

/// How far a value, or the control flow, may differ between invocations: one of the three
/// below, ordered (the most a value's operands differ, it differs).
pub type Level = u8;
/// The same in every invocation of the workgroup.
pub const WORKGROUP: Level = 0;
/// The same in every invocation of a subgroup; it may differ between subgroups.
pub const SUBGROUP: Level = 1;
/// It may differ between invocations.
pub const INVOCATION: Level = 2;

/// A collective operation in control flow less uniform than it needs.
#[derive(Clone, Debug, PartialEq)]
pub struct NonUniform {
    pub op: Collective,
    /// The function and the value the operation defines.
    pub func: FuncId,
    pub value: ValueId,
    /// Where the statement it's in comes from (the function's last `Stmt::At`).
    pub at: Option<Span>,
    /// The calls that reach `func`, from the entry point's function down: `path[0]` calls
    /// `path[1]`, and the last one calls `func`.
    pub path: Vec<FuncId>,
}

/// The collective operations (derivatives, samples, barriers, subgroup operations) the entry
/// point `entry` reaches in control flow less uniform than they need. Its parameters (a
/// fragment shader's varyings) differ between invocations, as do its builtin `inputs`.
pub fn collectives_in_non_uniform_flow(
    m: &Module,
    entry: FuncId,
    inputs: &[BuiltinInput],
) -> Vec<NonUniform> {
    let mut an =
        Analysis { m, inputs, memo: HashMap::new(), seen: HashSet::new(), out: Vec::new() };
    let params = vec![INVOCATION; m.functions[entry.index()].params.len()];
    an.call(entry, params, WORKGROUP, &mut Vec::new());
    an.out
}

/// A function's effect on uniformity for given inputs: how far its result, and each of its
/// parameters when it returns (a by-reference one may be written), may differ.
#[derive(Clone)]
struct Summary {
    ret: Level,
    params: Vec<Level>,
}

struct Analysis<'m> {
    m: &'m Module,
    inputs: &'m [BuiltinInput],
    /// Each function's summary by its parameters' uniformity and its caller's flow.
    memo: HashMap<(FuncId, Vec<Level>, Level), Summary>,
    /// Each finding once: by function, value and statement (a barrier defines no value).
    seen: HashSet<(FuncId, ValueId, Option<Span>)>,
    out: Vec<NonUniform>,
}

impl Analysis<'_> {
    fn call(&mut self, g: FuncId, args: Vec<Level>, cf: Level, path: &mut Vec<FuncId>) -> Summary {
        let key = (g, args.clone(), cf);
        if let Some(s) = self.memo.get(&key) {
            return s.clone();
        }
        let m = self.m;
        let f = &m.functions[g.index()];
        if path.contains(&g) {
            // Recursion, which GPU code reports itself (E0600): assume the worst.
            return Summary { ret: INVOCATION, params: vec![INVOCATION; args.len()] };
        }
        path.push(g);
        let mut st = State { locals: vec![cf; f.locals.len()], params: args };
        let mut fa = Fa {
            an: self,
            id: g,
            path,
            values: vec![WORKGROUP; f.values.len()],
            at: None,
            loops: Vec::new(),
            ret: WORKGROUP,
            ret_state: None,
        };
        let (beh, _) = fa.block(&f.body, cf, &mut st);
        if beh & NEXT != 0 {
            join(&mut fa.ret_state, &st);
        }
        let s = Summary { ret: fa.ret, params: fa.ret_state.unwrap_or(st).params };
        path.pop();
        self.memo.insert(key, s.clone());
        s
    }
}

/// How a statement can end (WGSL's behaviors): it goes on, or leaves by these.
type Behavior = u8;
const NEXT: Behavior = 1;
const BREAK: Behavior = 2;
const CONTINUE: Behavior = 4;
const RETURN: Behavior = 8;

/// How far what each local and parameter holds may differ between invocations.
#[derive(Clone, PartialEq)]
struct State {
    locals: Vec<Level>,
    params: Vec<Level>,
}

impl State {
    fn join(&mut self, o: &State) {
        for (a, b) in self.locals.iter_mut().zip(&o.locals) {
            *a = (*a).max(*b);
        }
        for (a, b) in self.params.iter_mut().zip(&o.params) {
            *a = (*a).max(*b);
        }
    }
}

/// Adds `s` to the states that meet at one point.
fn join(into: &mut Option<State>, s: &State) {
    match into {
        Some(x) => x.join(s),
        None => *into = Some(s.clone()),
    }
}

/// The states where a loop's iterations leave it, and go on to its next iteration.
#[derive(Default)]
struct LoopExits {
    breaks: Option<State>,
    continues: Option<State>,
    continue_cf: Level,
}

/// One function's analysis, for one summary.
struct Fa<'a, 'm> {
    an: &'a mut Analysis<'m>,
    id: FuncId,
    path: &'a mut Vec<FuncId>,
    /// How far each value may differ between invocations.
    values: Vec<Level>,
    at: Option<Span>,
    loops: Vec<LoopExits>,
    ret: Level,
    /// The state wherever it returns.
    ret_state: Option<State>,
}

impl Fa<'_, '_> {
    /// Runs a block in flow `cf`: how it ends, and the flow after it.
    fn block(&mut self, b: &[Stmt], cf: Level, st: &mut State) -> (Behavior, Level) {
        let (mut beh, mut cf) = (NEXT, cf);
        for s in b {
            if beh & NEXT == 0 {
                break;
            }
            let (sb, scf) = self.stmt(s, cf, st);
            beh = (beh & !NEXT) | sb;
            cf = scf;
        }
        (beh, cf)
    }

    fn stmt(&mut self, s: &Stmt, cf: Level, st: &mut State) -> (Behavior, Level) {
        match s {
            Stmt::Let(v, e) => {
                self.values[v.index()] = self.expr(*v, e, cf, st);
            }
            Stmt::Eval(e) => {
                self.expr(ValueId(u32::MAX), e, cf, st);
            }
            Stmt::Store(p, v) => {
                let t = self.values[v.index()].max(cf);
                self.store(p, t, st);
            }
            Stmt::If { cond, then, else_ } => {
                let inner = cf.max(self.values[cond.index()]);
                let mut a = st.clone();
                let (ba, cfa) = self.block(then, inner, &mut a);
                let mut b = st.clone();
                let (bb, cfb) = self.block(else_, inner, &mut b);
                let mut after: Option<State> = None;
                if ba & NEXT != 0 {
                    join(&mut after, &a);
                }
                if bb & NEXT != 0 {
                    join(&mut after, &b);
                }
                if let Some(after) = after {
                    *st = after;
                }
                let beh = ba | bb;
                // Branches that only fall through meet again.
                return (beh, if beh == NEXT { cf } else { cfa.max(cfb) });
            }
            Stmt::Loop { body, continuing } => return self.lp(body, continuing, cf, st),
            Stmt::Break => {
                if let Some(l) = self.loops.last_mut() {
                    join(&mut l.breaks, st);
                }
                return (BREAK, cf);
            }
            Stmt::Continue => {
                if let Some(l) = self.loops.last_mut() {
                    join(&mut l.continues, st);
                    l.continue_cf = l.continue_cf.max(cf);
                }
                return (CONTINUE, cf);
            }
            Stmt::Return(v) => {
                let v = v.map_or(WORKGROUP, |v| self.values[v.index()]);
                self.ret = self.ret.max(cf).max(v);
                join(&mut self.ret_state, st);
                return (RETURN, cf);
            }
            // On the GPU, a trap is a point valid code never reaches (the WGSL back end writes
            // nothing for it), so it changes no control flow.
            Stmt::Trap => {}
            Stmt::At(span) => self.at = Some(*span),
        }
        (NEXT, cf)
    }

    /// A loop, run until what may differ stops growing.
    fn lp(
        &mut self,
        body: &[Stmt],
        continuing: &[Stmt],
        cf: Level,
        st: &mut State,
    ) -> (Behavior, Level) {
        self.loops.push(LoopExits::default());
        let (mut head_cf, mut head) = (cf, st.clone());
        let beh = loop {
            let mut s = head.clone();
            let (bb, cfb) = self.block(body, head_cf, &mut s);
            let exits = self.loops.last_mut().expect("in a loop");
            let mut next: Option<State> = exits.continues.clone();
            let mut next_cf = exits.continue_cf;
            if bb & NEXT != 0 {
                join(&mut next, &s);
                next_cf = next_cf.max(cfb);
            }
            let mut again = (head_cf, head.clone());
            if let Some(mut c) = next {
                let (_, cfc) = self.block(continuing, next_cf, &mut c);
                again.0 = again.0.max(cfc);
                again.1.join(&c);
            }
            if again.0 == head_cf && again.1 == head {
                break bb;
            }
            (head_cf, head) = again;
        };
        let exits = self.loops.pop().unwrap_or_default();
        if let Some(b) = exits.breaks {
            *st = b;
        }
        let beh = (beh & RETURN) | if beh & BREAK != 0 { NEXT } else { 0 };
        // A loop that's only left by `break` ends with its iterations together again.
        (beh, if beh == NEXT { cf } else { head_cf })
    }

    fn place(&self, p: &Place, st: &State) -> Level {
        let root = match p.root {
            PlaceRoot::Local(l) => st.locals[l.index()],
            PlaceRoot::Param(i) => st.params[i as usize],
            PlaceRoot::Resource(r) => match self.an.m.resources[r.index()].kind {
                // Every invocation reads the same block, or buffer element at the same index.
                ResourceKind::Uniform { .. }
                | ResourceKind::StorageRead
                | ResourceKind::Texture { .. }
                | ResourceKind::StorageTexture { .. }
                | ResourceKind::Sampler { .. } => WORKGROUP,
                // Workgroup memory is written by other invocations (WGSL: non-uniform loads).
                ResourceKind::StorageReadWrite
                | ResourceKind::Private
                | ResourceKind::Workgroup => INVOCATION,
            },
            PlaceRoot::Ptr(v) => self.values[v.index()],
            // Constant data is the same for every invocation.
            PlaceRoot::Data(_) => WORKGROUP,
        };
        let mut index = WORKGROUP;
        p.for_each_value(&mut |v| index = index.max(self.values[v.index()]));
        root.max(index)
    }

    fn store(&self, p: &Place, t: Level, st: &mut State) {
        let mut t = t;
        p.for_each_value(&mut |v| t = t.max(self.values[v.index()]));
        let slot = match p.root {
            PlaceRoot::Local(l) => &mut st.locals[l.index()],
            PlaceRoot::Param(i) => &mut st.params[i as usize],
            PlaceRoot::Resource(_) | PlaceRoot::Ptr(_) | PlaceRoot::Data(_) => return,
        };
        // A store to part of it keeps what the rest may hold.
        *slot = if p.path.is_empty() { t } else { (*slot).max(t) };
    }

    /// Notes a collective operation `op` defining `v`, if it's reached in control flow less
    /// uniform than it needs.
    fn collective(&mut self, op: Collective, v: ValueId, cf: Level) {
        if cf > op.needs() && self.an.seen.insert((self.id, v, self.at)) {
            let mut path = self.path.clone();
            path.pop();
            let (func, at) = (self.id, self.at);
            self.an.out.push(NonUniform { op, func, value: v, at, path });
        }
    }

    /// How far the value `e` computes may differ between invocations; `v` is the value it
    /// defines.
    fn expr(&mut self, v: ValueId, e: &Expr, cf: Level, st: &mut State) -> Level {
        let operands = |fa: &Self| {
            let mut t = cf;
            e.for_each_value(&mut |x| t = t.max(fa.values[x.index()]));
            t
        };
        match e {
            Expr::Load(p) => cf.max(self.place(p, st)),
            Expr::EntryInput(i) => cf.max(match self.an.inputs.get(*i as usize) {
                Some(
                    BuiltinInput::WorkgroupId
                    | BuiltinInput::NumWorkgroups
                    | BuiltinInput::SubgroupSize,
                ) => WORKGROUP,
                _ => INVOCATION,
            }),
            Expr::Param(i) => cf.max(st.params[*i as usize]),
            Expr::Builtin(b @ (Builtin::Dpdx | Builtin::Dpdy | Builtin::Fwidth), _) => {
                let op = match b {
                    Builtin::Dpdx => Collective::Dpdx,
                    Builtin::Dpdy => Collective::Dpdy,
                    _ => Collective::Fwidth,
                };
                self.collective(op, v, cf);
                // Neighbouring pixels' values differ.
                INVOCATION
            }
            // A barrier: every invocation of the workgroup must reach it together.
            Expr::Barrier => {
                self.collective(Collective::Barrier, v, cf);
                WORKGROUP
            }
            // Every invocation of the subgroup must reach it together. A reduction or a
            // broadcast gives one value to the whole subgroup (another to another subgroup), as
            // does a shuffle from a place that's the same across the subgroup; a scan, or a
            // shuffle from places that differ, gives each invocation its own.
            Expr::Subgroup(op, args) => {
                self.collective(Collective::Subgroup(*op), v, cf);
                let one = match op {
                    SubgroupOp::ExclusiveAdd | SubgroupOp::InclusiveAdd => false,
                    SubgroupOp::Shuffle => {
                        args.get(1).is_some_and(|i| self.values[i.index()] <= SUBGROUP)
                    }
                    _ => true,
                };
                if one { cf.max(SUBGROUP) } else { INVOCATION }
            }
            // Another invocation may have changed what's there.
            Expr::Atomic(..) => INVOCATION,
            Expr::Texture(op, ..) if op.uses_derivatives() => {
                let op = match op {
                    TextureOp::Sample => Collective::Sample,
                    TextureOp::SampleBias => Collective::SampleBias,
                    _ => Collective::SampleCompare,
                };
                self.collective(op, v, cf);
                operands(self)
            }
            Expr::Call(g, args) => {
                let ins: Vec<Level> = args
                    .iter()
                    .map(|a| match a {
                        Arg::Value(x) => self.values[x.index()],
                        Arg::Place(p) => self.place(p, st),
                    })
                    .collect();
                let s = self.an.call(*g, ins, cf, self.path);
                for (a, &out) in args.iter().zip(&s.params) {
                    if let Arg::Place(p) = a {
                        self.store(p, out, st);
                    }
                }
                s.ret
            }
            _ => operands(self),
        }
    }
}
