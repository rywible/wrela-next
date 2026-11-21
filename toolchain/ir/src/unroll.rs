//! Unrolling (GPU): a loop that counts a local from a constant by a constant step and leaves
//! only when the count reaches a constant bound, a few times, is written out once per pass.
//!
//! Why: wgpu bounds every WGSL loop it translates for Metal (a counter that breaks out after
//! 2⁶⁴ passes, so a loop can't hang the GPU), and Metal's compiler then keeps the loop: the
//! temporal resolve's 2 × 2 reconstruction as a loop took 0.88 ms at 1080p, written out 0.53.
//! Written out here, each pass's index is a constant the shader compiler folds.
//!
//! A loop is unrolled when:
//! - it is counted (`crate::counted`) from a constant start, and has no `continue` of its own
//!   (a `break` of its own besides its test is kept: the copies are written inside a loop that
//!   runs once, so a `break` in any copy leaves after the copies before it, as the loop would);
//! - it runs at most `MOST_PASSES` times, and its copies come to at most `MOST_STATEMENTS`;
//! - it calls no function (those left are big ones, written out once per call by a browser's
//!   shader compiler: language.md §18.11's weight).
//!
//! The copies keep the counter's stores, so the counter holds what it would after the loop;
//! inner loops are unrolled first.

use crate::*;
use std::collections::{HashMap, HashSet};

/// The most passes a loop is unrolled for, and the most statements its copies may come to.
const MOST_PASSES: u32 = 16;
const MOST_STATEMENTS: usize = 2000;

/// Unrolls `f`'s loops that qualify (see the module docs).
pub(crate) fn unroll(m: &Module, f: &mut Function) {
    let mut consts = crate::counted::constants(f);
    // A count's limit written as constants added (`0..NB + 1`) is a constant too: this runs
    // before the module's constants are folded, so integer sums, differences and products of
    // constants are worked out here (exactly, as the GPU wraps them; never a float). One walk
    // finds them all: a value is defined before the code that reads it (`crate::verify`).
    visit::walk(&f.body, &mut |s| {
        if let Stmt::Let(v, Expr::Binary(op, a, b)) = s
            && !consts.contains_key(v)
            && let (Some(x), Some(y)) = (consts.get(a), consts.get(b))
            && let Some(c) = integer_op(*op, x, y)
        {
            consts.insert(*v, c);
        }
    });
    // The loops that leave on the count the command gives stay loops (W0014 says them), found in
    // the function as it is, before any is written out.
    let mut command = HashSet::new();
    let mut defs = None;
    visit::walk(&f.body, &mut |s| {
        if let Stmt::Loop { body, .. } = s
            && exits(body)
            && crate::rates::leaves_on_command(
                m,
                f,
                defs.get_or_insert_with(|| crate::rates::defs_and_stores(f)),
                body,
            )
        {
            command.insert(body.as_ptr() as usize);
        }
    });
    let body = std::mem::take(&mut f.body);
    f.body = block(f, body, &consts, &command);
}

fn block(
    f: &mut Function,
    b: Block,
    consts: &HashMap<ValueId, Const>,
    command: &HashSet<usize>,
) -> Block {
    let mut out: Block = Vec::with_capacity(b.len());
    for s in b {
        match s {
            Stmt::If { cond, then, else_ } => {
                let then = block(f, then, consts, command);
                let else_ = block(f, else_, consts, command);
                out.push(Stmt::If { cond, then, else_ });
            }
            Stmt::Loop { body, continuing } => {
                let on_command = command.contains(&(body.as_ptr() as usize));
                let body = block(f, body, consts, command);
                let continuing = block(f, continuing, consts, command);
                let planned = plan(&out, &body, &continuing, consts)
                    .filter(|&(_, _, leaves)| !(leaves && on_command));
                match planned {
                    Some((passes, check, leaves)) => {
                        let mut copies = Vec::new();
                        for _ in 0..passes {
                            let mut names = HashMap::new();
                            // The loop's own test (`if c >= limit { break }`, as `counted` found
                            // it) goes; any other `break` stays.
                            for (_, s) in body.iter().enumerate().filter(|&(i, _)| i != check) {
                                copies.push(copy(f, s, &mut names));
                            }
                            for s in &continuing {
                                copies.push(copy(f, s, &mut names));
                            }
                        }
                        if leaves {
                            // Its own `break`s set a flag, and what follows each in its pass and
                            // the passes after runs only while the flag is clear: nested `if`s,
                            // as a person writes a loop that may stop early out by hand (no
                            // loop is left for the GPU's compiler to keep).
                            let bool_ty = match &body[check] {
                                Stmt::If { cond, .. } => f.value_ty(*cond),
                                _ => unreachable!("the count's test is an `if`"),
                            };
                            let left = f.new_local("left", bool_ty);
                            let (no, yes) = (f.new_value(bool_ty), f.new_value(bool_ty));
                            out.push(Stmt::Let(no, Expr::Const(Const::Bool(false))));
                            out.push(Stmt::Let(yes, Expr::Const(Const::Bool(true))));
                            out.push(Stmt::Store(Place::local(left), no));
                            let copies: Vec<Stmt> =
                                copies.into_iter().map(|s| breaks_set(s, left, yes)).collect();
                            out.extend(nest(f, copies, left, bool_ty));
                        } else {
                            out.extend(copies);
                        }
                    }
                    None => out.push(Stmt::Loop { body, continuing }),
                }
            }
            other => out.push(other),
        }
    }
    out
}

/// `x op y` of two integer constants of one type, wrapping as the GPU does: a sum, a difference
/// or a product.
fn integer_op(op: BinOp, x: &Const, y: &Const) -> Option<Const> {
    let sum_or_product = matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul);
    let integers = matches!(x, Const::U32(_) | Const::I32(_));
    (sum_or_product && integers).then(|| crate::const_fold::binary(op, x, y)).flatten()
}

/// Whether `b` calls a function.
fn calls(b: &[Stmt]) -> bool {
    let mut found = false;
    visit::walk(b, &mut |s| found |= matches!(s.expr(), Some(Expr::Call(..))));
    found
}

/// How many times the loop runs at most, if it qualifies, its test's index in its body, and
/// whether it may leave sooner by a `break` of its own; `before` is the code before it in its
/// block.
fn plan(
    before: &[Stmt],
    body: &Block,
    continuing: &Block,
    consts: &HashMap<ValueId, Const>,
) -> Option<(u32, usize, bool)> {
    // A loop that calls a function stays a loop: the calls left after flattening are to big
    // functions (`opt::outlined`), which a browser's shader compiler copies into each place
    // they're called, so a copy per pass would cost far more to compile than the loop runs.
    if calls(body) || calls(continuing) {
        return None;
    }
    // Counted (`crate::counted`), from a known start, and leaving only at its test.
    let c = crate::counted::counted(consts, before, body, continuing)?;
    if continues(&body[c.check + 1..]) || exits(continuing) {
        return None;
    }
    let leaves = exits(&body[c.check + 1..]);
    let passes = c.passes?;
    let size = crate::opt::size_of(body) + crate::opt::size_of(continuing);
    if passes > u64::from(MOST_PASSES) || passes as usize * size > MOST_STATEMENTS {
        return None;
    }
    Some((passes as u32, c.check, leaves))
}

/// Whether code leaves the loop it's in by its own `break`, `continue` (not an inner loop's).
fn exits(b: &[Stmt]) -> bool {
    b.iter().any(|s| match s {
        Stmt::Break | Stmt::Continue => true,
        Stmt::If { then, else_, .. } => exits(then) || exits(else_),
        _ => false,
    })
}

/// `s` with each `break` of the loop it's in (not an inner loop's) a store of `yes` to `left`.
fn breaks_set(s: Stmt, left: LocalId, yes: ValueId) -> Stmt {
    match s {
        Stmt::Break => Stmt::Store(Place::local(left), yes),
        Stmt::If { cond, then, else_ } => Stmt::If {
            cond,
            then: then.into_iter().map(|s| breaks_set(s, left, yes)).collect(),
            else_: else_.into_iter().map(|s| breaks_set(s, left, yes)).collect(),
        },
        other => other,
    }
}

/// `b` with what follows each statement that may set `left` run only while it's clear: the
/// statement, then `if !left { the rest }`, nested.
fn nest(f: &mut Function, b: Vec<Stmt>, left: LocalId, bool_ty: TypeId) -> Vec<Stmt> {
    let mut out = Vec::new();
    let mut rest = b.into_iter();
    while let Some(s) = rest.next() {
        let sets = sets_local(&s, left);
        out.push(s);
        if sets {
            let tail: Vec<Stmt> = rest.collect();
            if tail.is_empty() {
                return out;
            }
            let (read, clear) = (f.new_value(bool_ty), f.new_value(bool_ty));
            out.push(Stmt::Let(read, Expr::Load(Place::local(left))));
            out.push(Stmt::Let(clear, Expr::Unary(UnOp::Not, read)));
            let then = nest(f, tail, left, bool_ty);
            out.push(Stmt::If { cond: clear, then, else_: Vec::new() });
            return out;
        }
    }
    out
}

/// Whether `s` stores to `left` outside any loop of its own.
fn sets_local(s: &Stmt, left: LocalId) -> bool {
    match s {
        Stmt::Store(p, _) => p.root_local() == Some(left),
        Stmt::If { then, else_, .. } => {
            then.iter().any(|s| sets_local(s, left)) || else_.iter().any(|s| sets_local(s, left))
        }
        _ => false,
    }
}

/// Whether code goes on to the next pass of the loop it's in by its own `continue`.
fn continues(b: &[Stmt]) -> bool {
    b.iter().any(|s| match s {
        Stmt::Continue => true,
        Stmt::If { then, else_, .. } => continues(then) || continues(else_),
        _ => false,
    })
}

/// A copy of `s` whose values are new: each value it defines gets a new name, in `names`, and
/// the values it uses go by theirs.
fn copy(f: &mut Function, s: &Stmt, names: &mut HashMap<ValueId, ValueId>) -> Stmt {
    // Copy control flow's children once. Cloning the whole statement before copying each
    // child again would copy every nested block once for each block around it.
    let mut out = match s {
        Stmt::If { cond, .. } => Stmt::If { cond: *cond, then: Vec::new(), else_: Vec::new() },
        Stmt::Loop { .. } => Stmt::Loop { body: Vec::new(), continuing: Vec::new() },
        other => other.clone(),
    };
    out.for_each_value_mut(&mut |x| {
        if let Some(n) = names.get(x) {
            *x = *n;
        }
    });
    match (s, &mut out) {
        (_, Stmt::Let(v, _)) => {
            let n = f.new_value(f.value_ty(*v));
            names.insert(*v, n);
            *v = n;
        }
        (Stmt::If { then, else_, .. }, Stmt::If { then: out_then, else_: out_else, .. }) => {
            *out_then = then.iter().map(|s| copy(f, s, names)).collect();
            *out_else = else_.iter().map(|s| copy(f, s, names)).collect();
        }
        (Stmt::Loop { body, continuing }, Stmt::Loop { body: out_body, continuing: out_cont }) => {
            *out_body = body.iter().map(|s| copy(f, s, names)).collect();
            *out_cont = continuing.iter().map(|s| copy(f, s, names)).collect();
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn old_copy(
        f: &mut Function,
        s: &Stmt,
        names: &mut HashMap<ValueId, ValueId>,
        cloned: &mut usize,
    ) -> Stmt {
        visit::walk(std::slice::from_ref(s), &mut |_| *cloned += 1);
        let mut s = s.clone();
        s.for_each_value_mut(&mut |x| {
            if let Some(n) = names.get(x) {
                *x = *n;
            }
        });
        match &mut s {
            Stmt::Let(v, _) => {
                let n = f.new_value(f.value_ty(*v));
                names.insert(*v, n);
                *v = n;
            }
            Stmt::If { then, else_, .. } => {
                *then = then.iter().map(|s| old_copy(f, s, names, cloned)).collect();
                *else_ = else_.iter().map(|s| old_copy(f, s, names, cloned)).collect();
            }
            Stmt::Loop { body, continuing } => {
                *body = body.iter().map(|s| old_copy(f, s, names, cloned)).collect();
                *continuing = continuing.iter().map(|s| old_copy(f, s, names, cloned)).collect();
            }
            _ => {}
        }
        s
    }

    /// `i = 0; loop { if i >= 5 { break }; if acc == 7 { break }; acc = i } continuing { i = i
    /// + 1 }`, its second exit's statement `leave` (a `break`, or a `continue` to keep it a loop).
    fn leaves_early(leave: Stmt) -> Function {
        let mut types = Types::default();
        let (u, b) = (types.u32(), types.bool());
        let mut f = Function::new("early", Vec::new(), None);
        let (i, acc) = (f.new_local("i", u), f.new_local("acc", u));
        let (start, c, limit, t) = (f.new_value(u), f.new_value(u), f.new_value(u), f.new_value(b));
        let (a, seven, hit) = (f.new_value(u), f.new_value(u), f.new_value(b));
        let (r, one, n) = (f.new_value(u), f.new_value(u), f.new_value(u));
        let body = vec![
            Stmt::Let(c, Expr::Load(Place::local(i))),
            Stmt::Let(limit, Expr::Const(Const::U32(5))),
            Stmt::Let(t, Expr::Binary(BinOp::Ge, c, limit)),
            Stmt::If { cond: t, then: vec![Stmt::Break], else_: vec![] },
            Stmt::Let(a, Expr::Load(Place::local(acc))),
            Stmt::Let(seven, Expr::Const(Const::U32(7))),
            Stmt::Let(hit, Expr::Binary(BinOp::Eq, a, seven)),
            Stmt::If { cond: hit, then: vec![leave], else_: vec![] },
            Stmt::Store(Place::local(acc), c),
        ];
        let continuing = vec![
            Stmt::Let(r, Expr::Load(Place::local(i))),
            Stmt::Let(one, Expr::Const(Const::U32(1))),
            Stmt::Let(n, Expr::Binary(BinOp::Add, r, one)),
            Stmt::Store(Place::local(i), n),
        ];
        f.body = vec![
            Stmt::Let(start, Expr::Const(Const::U32(0))),
            Stmt::Store(Place::local(i), start),
            Stmt::Loop { body, continuing },
        ];
        f
    }

    /// The tile kernel spike's fold (a loop over a part's members that leaves at the first
    /// missing one) took 2.6x its written-out form when it stayed a loop.
    #[test]
    fn a_counted_loop_that_leaves_early_is_written_out_as_nested_ifs() {
        let mut f = leaves_early(Stmt::Break);
        unroll(&Module::default(), &mut f);
        // No loop is left: each copy's early `break` sets a flag, and what follows it runs
        // under `if !flag`, nested: four `if`s deep (the last copy's rest is empty).
        let mut loops = 0;
        visit::walk(&f.body, &mut |s| loops += usize::from(matches!(s, Stmt::Loop { .. })));
        assert_eq!(loops, 0, "{:?}", f.body);
        fn deepest(b: &[Stmt]) -> usize {
            b.iter()
                .map(|s| match s {
                    Stmt::If { then, else_, .. } => 1 + deepest(then).max(deepest(else_)),
                    _ => 0,
                })
                .max()
                .unwrap_or(0)
        }
        // Four `if !flag` levels (after passes 1 to 4), and the fifth pass's own early test.
        assert_eq!(deepest(&f.body), 5, "{:?}", f.body);
    }

    #[test]
    fn a_counted_loop_with_a_continue_of_its_own_stays_a_loop() {
        let mut f = leaves_early(Stmt::Continue);
        let before = f.body.clone();
        unroll(&Module::default(), &mut f);
        assert_eq!(f.body, before);
    }

    #[test]
    fn codegen_perf_unroll_copies_nested_control_flow_once() {
        let mut types = Types::default();
        let (u, b) = (types.u32(), types.bool());
        let mut f = Function::new("nested", Vec::new(), None);
        let local = f.new_local("counter", u);
        let (cond, value) = (f.new_value(b), f.new_value(u));
        let mut nested = Stmt::Store(Place::local(local), value);
        for _ in 0..12 {
            let next = f.new_value(u);
            nested = Stmt::If {
                cond,
                then: vec![Stmt::Let(next, Expr::Binary(BinOp::Add, value, value)), nested],
                else_: vec![Stmt::Store(Place::local(local), value)],
            };
        }
        let source = Stmt::Loop {
            body: vec![
                Stmt::Let(cond, Expr::Const(Const::Bool(true))),
                Stmt::Let(value, Expr::Const(Const::U32(3))),
                nested,
            ],
            continuing: vec![Stmt::Store(Place::local(local), value)],
        };
        let mut expected_f = f.clone();
        let (mut names, mut expected_names) = (HashMap::new(), HashMap::new());
        let mut cloned = 0;
        let expected = old_copy(&mut expected_f, &source, &mut expected_names, &mut cloned);
        let actual = copy(&mut f, &source, &mut names);
        assert_eq!(actual, expected);
        assert_eq!(f, expected_f);
        assert_eq!(names, expected_names);
        let mut statements = 0;
        visit::walk(std::slice::from_ref(&source), &mut |_| statements += 1);
        assert_eq!(statements, 41);
        assert_eq!(cloned, 315);
    }
}
