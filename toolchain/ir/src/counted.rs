//! A counted loop: the one definition the passes that treat one specially read (`unroll`, which
//! writes it out, `hoist`, which moves what its passes compute alike onto the CPU, and the MSL
//! writer, which writes it as a `for`).
//!
//! A `Loop` is counted where a local (its counter) is left at the loop's first test and stepped
//! only by its `continuing`:
//!
//! - its body begins with pure `let`s, then `if c >= limit { break }` (or `if !(c < limit)`),
//!   `c` a read of the whole counter and `limit` a constant;
//! - its `continuing` holds pure `let`s and one store: the counter's read plus a constant `step`,
//!   more than 0, of the limit's type;
//! - nothing in its body stores the counter, writes it in part or lends it (its address, or a
//!   place a call is given).
//!
//! Its start is the constant stored to the counter just before the loop, where one is (nothing
//! between mentions the counter): then its passes are known.

use crate::*;
use std::collections::HashMap;

/// A counted loop's facts.
#[derive(Clone, Debug, PartialEq)]
pub struct Counted {
    pub local: LocalId,
    /// The constant it starts at, where the statements before the loop say.
    pub start: Option<Const>,
    pub limit: Const,
    pub step: Const,
    /// The exit test's index in the body.
    pub check: usize,
    /// The test's own values: the counter's read, the comparison (and its `!`), each with its
    /// statement's index in the body.
    pub test: Vec<(ValueId, usize)>,
    /// How many times its body runs, where its start is known and it doesn't leave sooner.
    pub passes: Option<u64>,
}

/// The constants `f` defines, by value.
pub fn constants(f: &Function) -> HashMap<ValueId, Const> {
    let mut consts = HashMap::new();
    visit::walk(&f.body, &mut |s| {
        if let Stmt::Let(v, Expr::Const(c)) = s {
            consts.insert(*v, c.clone());
        }
    });
    consts
}

/// [`Counted`] of the loop `body`/`continuing`, `before` the statements ahead of it in its block
/// and `consts` its function's constants ([`constants`]), if it is one.
pub fn counted(
    consts: &HashMap<ValueId, Const>,
    before: &[Stmt],
    body: &[Stmt],
    continuing: &[Stmt],
) -> Option<Counted> {
    // The body's head: pure `let`s, then the exit test.
    let mut defs: HashMap<ValueId, (&Expr, usize)> = HashMap::new();
    let mut check = None;
    for (i, s) in body.iter().enumerate() {
        match s {
            Stmt::At(_) => {}
            Stmt::Let(v, e) if pure(e) => {
                defs.insert(*v, (e, i));
            }
            Stmt::If { cond, then, else_ } if leaves(then) && only_at(else_) => {
                check = Some((i, *cond));
                break;
            }
            _ => return None,
        }
    }
    let (check, cond) = check?;
    let mut test = Vec::new();
    // `c >= limit`, or `!(c < limit)`.
    let &(cond_e, cond_at) = defs.get(&cond)?;
    let (read, n) = match cond_e {
        Expr::Binary(BinOp::Ge, x, n) => {
            test.push((cond, cond_at));
            (*x, *n)
        }
        Expr::Unary(UnOp::Not, y) => {
            let &(inner, at) = defs.get(y)?;
            let Expr::Binary(BinOp::Lt, x, n) = inner else { return None };
            test.push((cond, cond_at));
            test.push((*y, at));
            (*x, *n)
        }
        _ => return None,
    };
    let &(read_e, read_at) = defs.get(&read)?;
    let Expr::Load(p) = read_e else { return None };
    let local = p.whole_local()?;
    test.push((read, read_at));
    let limit = consts.get(&n).filter(|c| int(c)).cloned()?;
    // The step: `continuing` holds pure `let`s and the counter's one store, its read plus a
    // constant.
    let mut cdefs: HashMap<ValueId, &Expr> = HashMap::new();
    let mut step = None;
    for s in continuing {
        match s {
            Stmt::At(_) => {}
            Stmt::Let(v, e) if pure(e) => {
                cdefs.insert(*v, e);
            }
            Stmt::Store(p, v) if p.whole_local() == Some(local) && step.is_none() => {
                let Some(Expr::Binary(BinOp::Add, a, s)) = cdefs.get(v) else { return None };
                if !matches!(cdefs.get(a), Some(Expr::Load(q)) if q.whole_local() == Some(local)) {
                    return None;
                }
                step = Some(consts.get(s).cloned()?);
            }
            _ => return None,
        }
    }
    let step = step?;
    if std::mem::discriminant(&limit) != std::mem::discriminant(&step)
        || matches!(step, Const::U32(0) | Const::I32(i32::MIN..=0))
    {
        return None;
    }
    // Nothing in the body stores the counter, writes it in part or lends it.
    let mut touched = false;
    visit::walk(body, &mut |s| {
        if let Stmt::Store(p, _) = s {
            touched |= p.root_local() == Some(local);
        }
        if let Some(e) = s.expr() {
            let lends = matches!(e, Expr::Addr(_) | Expr::Call(..) | Expr::Atomic(..));
            e.for_each_place(&mut |p| touched |= lends && p.root_local() == Some(local));
        }
    });
    if touched {
        return None;
    }
    // Its start: the last store of it before the loop, nothing between mentioning it.
    let mut start = None;
    for s in before.iter().rev() {
        if let Stmt::Store(p, v) = s
            && p.whole_local() == Some(local)
        {
            start = consts.get(v).filter(|c| int(c)).cloned();
            break;
        }
        if mentions(s, local) {
            break;
        }
    }
    let passes = start.as_ref().and_then(|s| passes(s, &limit, &step));
    test.sort_by_key(|&(_, at)| at);
    Some(Counted { local, start, limit, step, check, test, passes })
}

/// How many passes from `start` below `limit` by `step` (all of one integer type).
fn passes(start: &Const, limit: &Const, step: &Const) -> Option<u64> {
    let (s, l, k) = match (start, limit, step) {
        (Const::U32(s), Const::U32(l), Const::U32(k)) => {
            (i64::from(*s), i64::from(*l), i64::from(*k))
        }
        (Const::I32(s), Const::I32(l), Const::I32(k)) => {
            (i64::from(*s), i64::from(*l), i64::from(*k))
        }
        _ => return None,
    };
    if s >= l {
        return Some(0);
    }
    // The counter's last value, `s + (passes - 1) * k`, mustn't wrap past its type's top: a step
    // past it wraps (u32) or overflows (i32) where the loop's `<` would read it otherwise.
    let n = (l - s + k - 1) / k;
    let top =
        if matches!(start, Const::U32(_)) { i64::from(u32::MAX) } else { i64::from(i32::MAX) };
    (s + n * k <= top).then_some(n as u64)
}

/// What a test's or a step's `let` may compute: reads, constants, arithmetic, comparisons.
fn pure(e: &Expr) -> bool {
    matches!(e, Expr::Load(_) | Expr::Const(_) | Expr::Binary(..) | Expr::Unary(..))
}

fn int(c: &Const) -> bool {
    matches!(c, Const::U32(_) | Const::I32(_))
}

/// `break` and source locations alone.
fn leaves(b: &[Stmt]) -> bool {
    let mut breaks = 0;
    for s in b {
        match s {
            Stmt::At(_) => {}
            Stmt::Break => breaks += 1,
            _ => return false,
        }
    }
    breaks == 1
}

fn only_at(b: &[Stmt]) -> bool {
    b.iter().all(|s| matches!(s, Stmt::At(_)))
}

fn mentions(s: &Stmt, l: LocalId) -> bool {
    visit::any(std::slice::from_ref(s), &mut |s| {
        let mut hit = false;
        s.for_each_place(&mut |p| hit |= p.root_local() == Some(l));
        hit
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `i = start; loop { c = i; n = limit; t = c >= n; if t { break } ; ... } continuing {
    /// a = i; s = step; b = a + s; i = b }`, with the step's constant in `continuing`.
    fn counted_loop(start: Const, limit: Const, step: Const) -> (HashMap<ValueId, Const>, Block) {
        let l = LocalId(0);
        let v = ValueId;
        let mut consts = HashMap::new();
        consts.insert(v(0), start.clone());
        consts.insert(v(2), limit.clone());
        consts.insert(v(6), step.clone());
        let body = vec![
            Stmt::Let(v(1), Expr::Load(Place::local(l))),
            Stmt::Let(v(2), Expr::Const(limit)),
            Stmt::Let(v(3), Expr::Binary(BinOp::Ge, v(1), v(2))),
            Stmt::If { cond: v(3), then: vec![Stmt::Break], else_: vec![] },
        ];
        let continuing = vec![
            Stmt::Let(v(5), Expr::Load(Place::local(l))),
            Stmt::Let(v(6), Expr::Const(step)),
            Stmt::Let(v(7), Expr::Binary(BinOp::Add, v(5), v(6))),
            Stmt::Store(Place::local(l), v(7)),
        ];
        let block = vec![
            Stmt::Let(v(0), Expr::Const(start)),
            Stmt::Store(Place::local(l), v(0)),
            Stmt::Loop { body, continuing },
        ];
        (consts, block)
    }

    fn of(consts: &HashMap<ValueId, Const>, b: &Block) -> Option<Counted> {
        let Stmt::Loop { body, continuing } = &b[2] else { unreachable!() };
        counted(consts, &b[..2], body, continuing)
    }

    #[test]
    fn a_loop_with_its_steps_constant_in_its_continuing_is_counted() {
        let (consts, b) = counted_loop(Const::I32(0), Const::I32(14), Const::I32(1));
        let c = of(&consts, &b).expect("counted");
        assert_eq!((c.local, c.passes, c.check), (LocalId(0), Some(14), 3));
        assert_eq!(c.test, vec![(ValueId(1), 0), (ValueId(3), 2)]);
    }

    #[test]
    fn passes_round_up_and_a_step_past_the_top_is_not_known() {
        let (consts, b) = counted_loop(Const::U32(1), Const::U32(10), Const::U32(4));
        assert_eq!(of(&consts, &b).unwrap().passes, Some(3));
        let (consts, b) =
            counted_loop(Const::U32(0), Const::U32(u32::MAX), Const::U32(u32::MAX - 1));
        assert_eq!(of(&consts, &b).unwrap().passes, None);
        let (consts, b) = counted_loop(Const::U32(5), Const::U32(5), Const::U32(1));
        assert_eq!(of(&consts, &b).unwrap().passes, Some(0));
    }

    #[test]
    fn a_body_that_stores_its_counter_or_a_step_of_zero_is_not_counted() {
        let (consts, mut b) = counted_loop(Const::U32(0), Const::U32(8), Const::U32(1));
        if let Stmt::Loop { body, .. } = &mut b[2] {
            body.push(Stmt::Store(Place::local(LocalId(0)), ValueId(0)));
        }
        assert_eq!(of(&consts, &b), None);
        let (consts, b) = counted_loop(Const::U32(0), Const::U32(8), Const::U32(0));
        assert_eq!(of(&consts, &b), None);
    }
}
