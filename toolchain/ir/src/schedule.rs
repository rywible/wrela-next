//! A frame's schedule, as the build makes it (docs/plans/engine-at-build.md phase 4 step 4): the
//! passes a function records, in the order its code records them, each a label
//! (`std::gpu::label`) and the pipelines its dispatches and draws record until the next label.
//! It's read from the CPU module after inlining and the folding of constant branches, so a pass
//! that a constant branch leaves out (one no kind of the scene uses) isn't in it, as its
//! pipelines aren't in the build. A command under a branch the build couldn't settle is
//! conditional: whether it runs is the frame's (the eye's, the game's, the GPU's counts).
//!
//! The walk follows each call in order (a function called again is walked again, as its commands
//! are recorded again); a call back into a function being walked is a recursion, not walked
//! twice. A label's text is the constant data its argument points at, followed through the
//! parameters it's passed as.

use crate::*;
use std::collections::HashMap;

/// The opcode of a label command (`wrela_abi::stream::Opcode::Label`).
const LABEL: u32 = wrela_abi::stream::Opcode::Label as u32;

/// One pass of the schedule: its label (`""` before the frame's first) and the pipelines it
/// records, in the order it first records each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pass {
    pub label: String,
    /// Whether the label itself is under a branch the build couldn't settle.
    pub conditional: bool,
    pub commands: Vec<Command>,
}

/// A pipeline a pass records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Command {
    pub pipeline: u32,
    pub draw: bool,
    /// Whether every place the pass records it is under a branch the build couldn't settle.
    pub conditional: bool,
    /// Whether any place the pass records it is in a loop.
    pub in_loop: bool,
}

/// The schedule of what `entry` records.
pub fn schedule(m: &Module, entry: FuncId) -> Vec<Pass> {
    let mut w = Walk {
        m,
        defs: HashMap::new(),
        stack: Vec::new(),
        passes: vec![Pass { label: String::new(), conditional: false, commands: Vec::new() }],
    };
    w.function(entry, &[], false, false);
    w.passes.retain(|p| !p.label.is_empty() || !p.commands.is_empty());
    w.passes
}

struct Walk<'a> {
    m: &'a Module,
    /// Each function's `let`s by the value they define.
    defs: HashMap<FuncId, HashMap<ValueId, &'a Expr>>,
    stack: Vec<FuncId>,
    passes: Vec<Pass>,
}

impl<'a> Walk<'a> {
    fn defs_of(&mut self, f: FuncId) {
        if self.defs.contains_key(&f) {
            return;
        }
        let mut d = HashMap::new();
        visit::walk(&self.m.functions[f.index()].body, &mut |s| {
            if let Stmt::Let(v, e) = s {
                d.insert(*v, e);
            }
        });
        self.defs.insert(f, d);
    }

    /// The text that value `v` of function `f` points at, where it's constant data: through a
    /// run built of a data's address and a length, and through a parameter to what the caller
    /// passed (`params`).
    fn text(&self, f: FuncId, v: ValueId, params: &[Option<String>]) -> Option<String> {
        let d = &self.defs[&f];
        let data = |p: &Place| match p.root {
            PlaceRoot::Data(id) if p.path.is_empty() => match &self.m.data[id.index()].value {
                ConstValue::Bytes(b) => Some(String::from_utf8_lossy(b).into_owned()),
                _ => None,
            },
            _ => None,
        };
        match d.get(&v)? {
            Expr::Param(i) => params.get(*i as usize).cloned().flatten(),
            Expr::Run(p) | Expr::Addr(p) => data(p),
            Expr::Construct(_, parts) => self.text(f, *parts.first()?, params),
            Expr::Mem(MemOp::Addr, args) => self.text(f, *args.first()?, params),
            _ => None,
        }
    }

    fn function(&mut self, f: FuncId, params: &[Option<String>], cond: bool, in_loop: bool) {
        if self.stack.contains(&f) {
            return;
        }
        self.defs_of(f);
        self.stack.push(f);
        let body = &self.m.functions[f.index()].body;
        self.block(f, body, params, cond, in_loop);
        self.stack.pop();
    }

    fn block(&mut self, f: FuncId, b: &'a [Stmt], params: &[Option<String>], cond: bool, lp: bool) {
        for s in b {
            match s {
                Stmt::If { then, else_, .. } => {
                    self.block(f, then, params, true, lp);
                    self.block(f, else_, params, true, lp);
                }
                Stmt::Loop { body, continuing } => {
                    self.block(f, body, params, cond, true);
                    self.block(f, continuing, params, cond, true);
                }
                Stmt::Let(_, e) | Stmt::Eval(e) => self.expr(f, e, params, cond, lp),
                _ => {}
            }
        }
    }

    fn expr(&mut self, f: FuncId, e: &'a Expr, params: &[Option<String>], cond: bool, lp: bool) {
        match e {
            Expr::Call(g, args) => {
                let passed: Vec<Option<String>> = args
                    .iter()
                    .map(|a| match a {
                        Arg::Value(v) => self.text(f, *v, params),
                        Arg::Place(_) => None,
                    })
                    .collect();
                self.function(*g, &passed, cond, lp);
            }
            Expr::Host(HostOp::Command { opcode, .. }, args) if *opcode == LABEL => {
                let label = args.first().and_then(|&v| self.text(f, v, params));
                self.passes.push(Pass {
                    label: label.unwrap_or_else(|| "(a label the build can't read)".into()),
                    conditional: cond,
                    commands: Vec::new(),
                });
            }
            Expr::Host(HostOp::Dispatch { pipeline, .. }, _) => {
                self.command(*pipeline, false, cond, lp)
            }
            Expr::Host(HostOp::Draw { pipeline, .. }, _) => self.command(*pipeline, true, cond, lp),
            _ => {}
        }
    }

    fn command(&mut self, pipeline: u32, draw: bool, cond: bool, lp: bool) {
        let pass = self.passes.last_mut().expect("the first pass is made before the walk");
        match pass.commands.iter_mut().find(|c| c.pipeline == pipeline) {
            Some(c) => {
                c.conditional &= cond;
                c.in_loop |= lp;
            }
            None => pass.commands.push(Command { pipeline, draw, conditional: cond, in_loop: lp }),
        }
    }
}
