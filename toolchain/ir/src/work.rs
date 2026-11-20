//! Work (GPU, language.md §18.11): the most work one call of a function can do, counted from its
//! code before the shader compiler's own optimizations, in units of about one 32-bit operation
//! on one invocation. A count is the same on every machine, so a build can check it
//! (`@budget(work: n)`); it is an upper bound to compare code by, not a time.
//!
//! The count walks the code as it would run, knowing only what the code makes constant: its
//! constants, `const` parameters folded into a pipeline, and the values computed from them. So
//! a branch on a constant counts the side it takes; a branch on anything else counts the costlier
//! side; and a loop counts each pass it makes until an exit it's sure of (a `for` over a constant
//! range, a `while` whose counter reaches a constant). A loop with no such exit within
//! [`MOST_PASSES`] passes is unbounded, and so is the work of what holds it.
//!
//! A `break` or `continue` on a condition the count doesn't know is a way out it follows too
//! ([`Exits`]): what's known after a loop is what every `break` of every pass agrees on (of an
//! integer, the most any leaves it), and on the next pass what the pass's end and each
//! `continue` agree on. A bound may be loose; it must never be low, since the host's step guard
//! trusts it. (Keeping only the side that went on once took the opening's cloud march, whose
//! inner loop may leave early, as leaving its outer loop on the first pass: 2,498 work for
//! 957,188.)
//!
//! Of a 32-bit integer it doesn't know, the count may still know the most it can be: a clamp
//! (`min(n, 64)`, `clamp(n, 0, 64)`), a mask (`n & 63`), a remainder by a constant (`n % 64`),
//! or what's made of those by addition, multiplication, division and shifts that can't wrap.
//! So a loop over a count read at run time is bounded by the clamp an author puts on it: `for i
//! in 0..min(count, 64)` makes at most 64 passes, as the count knows when `i` reaches 64.
//!
//! What each operation counts is [`cost`]'s table: arithmetic one a component, a division or a
//! transcendental four, a read of a storage buffer eight, a texture sample sixteen.

use crate::*;
use std::collections::HashMap;
use wrela_diag::Span;

/// The most passes a loop is followed for: past it, the loop's work is unbounded.
pub const MOST_PASSES: u32 = 1 << 12;

/// The most statements the count follows for one entry point (a guard on the count's own time):
/// past it, the work is unbounded.
pub const MOST_STEPS: u64 = 1 << 24;

/// Work units: arithmetic on one component.
pub const OP: u64 = 1;
/// A division or remainder, a square root or a transcendental, on one component.
pub const SLOW_OP: u64 = 4;
/// A read or a write of a storage buffer.
pub const MEMORY: u64 = 8;
/// A read or a write of workgroup memory.
pub const SHARED: u64 = 2;
/// A texture sample or gather.
pub const SAMPLE: u64 = 16;
/// A texel load or a storage texture's write.
pub const TEXEL: u64 = 8;
/// An atomic operation.
pub const ATOMIC: u64 = 16;

/// A function over its budget: where its `@budget` is, the budget, its work (`None`:
/// unbounded, at `unbounded`'s loop), and the calls that do the most of it.
#[derive(Clone, Debug, PartialEq)]
pub struct Breach {
    pub function: String,
    pub at: Span,
    pub budget: u64,
    pub work: Option<u64>,
    pub unbounded: Option<Span>,
    /// The functions it calls itself, by name, with their work, most first (at most three).
    pub biggest: Vec<(String, u64)>,
}

/// An entry point's work: `None` when unbounded, with the loop that's unbounded if one is.
#[derive(Clone, Debug, PartialEq)]
pub struct EntryWork {
    pub work: Option<u64>,
    pub unbounded: Option<Span>,
    /// Each function over its `@budget` in this module, once each (the largest call of it).
    pub breaches: Vec<Breach>,
}

/// The work of each entry point of module `m`, in its order, and the functions over their
/// budgets (once each). One count serves the module, so a function two entry points call is
/// followed once for each set of known arguments.
pub fn module_work(m: &Module) -> (Vec<EntryWork>, Vec<Breach>) {
    let mut c = Counter {
        m,
        steps: 0,
        gave_up: false,
        bounds: true,
        memo: HashMap::new(),
        breaches: Vec::new(),
        passes: HashMap::new(),
        proof: None,
        totals: None,
    };
    let mut out = Vec::new();
    for e in &m.entry_points {
        (c.steps, c.gave_up) = (0, false);
        let r = c.call(e.function, &[]);
        let work = if r.unbounded || c.gave_up { None } else { Some(r.work) };
        out.push(EntryWork { work, unbounded: r.loop_at, breaches: Vec::new() });
    }
    (out, c.breaches)
}

/// Each loop's most work, all its passes together, by its body's address (`as_ptr`), as the
/// count of each of module `m`'s entry points finds it (the host's bound, [`module_work`]'s):
/// `None` for a loop with no bound on some way the count followed. A loop the count never
/// reached (behind a constant branch, or after it gave up) isn't there. What fuel
/// (`crate::fuel`) leaves uncounted.
pub fn loop_totals(m: &Module) -> HashMap<usize, Option<u64>> {
    let mut c = Counter {
        m,
        steps: 0,
        gave_up: false,
        bounds: true,
        memo: HashMap::new(),
        breaches: Vec::new(),
        passes: HashMap::new(),
        proof: None,
        totals: Some(HashMap::new()),
    };
    for e in &m.entry_points {
        (c.steps, c.gave_up) = (0, false);
        c.call(e.function, &[]);
        if c.gave_up {
            // What it found after giving up isn't the code's: none of it is kept as a bound.
            return HashMap::new();
        }
    }
    c.totals.unwrap_or_default()
}

/// The work of entry point `f` of module `m`, and the functions over their budgets.
pub fn entry_work(m: &Module, f: FuncId) -> EntryWork {
    let mut c = Counter {
        m,
        steps: 0,
        gave_up: false,
        bounds: true,
        memo: HashMap::new(),
        breaches: Vec::new(),
        passes: HashMap::new(),
        proof: None,
        totals: None,
    };
    let r = c.call(f, &[]);
    let work = if r.unbounded || c.gave_up { None } else { Some(r.work) };
    EntryWork { work, unbounded: r.loop_at, breaches: c.breaches }
}

/// The work of single expressions of a module's functions, knowing nothing of their values (a
/// call's is its function's, given nothing of its arguments). One count serves each call, so
/// a function called many times is followed once.
pub struct ExprWork<'a> {
    c: Counter<'a>,
}

impl<'a> ExprWork<'a> {
    pub fn new(m: &'a Module) -> ExprWork<'a> {
        ExprWork {
            c: Counter {
                m,
                steps: 0,
                gave_up: false,
                bounds: false,
                memo: HashMap::new(),
                breaches: Vec::new(),
                passes: HashMap::new(),
                proof: None,
                totals: None,
            },
        }
    }

    /// The most work `e`, in `f`, can do: `None` when unbounded.
    pub fn of(&mut self, f: &Function, e: &Expr) -> Option<u64> {
        let mut fr = Frame {
            f,
            args: Vec::new(),
            vals: HashMap::new(),
            most: HashMap::new(),
            parts: HashMap::new(),
            nots: HashMap::new(),
            work: 0,
            unbounded: false,
            loop_at: None,
            here: None,
            ret: None,
            calls: HashMap::new(),
            sites: None,
            exits: Vec::new(),
            left: 0,
        };
        (self.c.steps, self.c.gave_up) = (0, false);
        self.c.expr(&mut fr, &mut State::default(), e);
        (!fr.unbounded && !self.c.gave_up).then_some(fr.work)
    }
}

/// The most passes each loop of function `f` makes, by its body's address (`as_ptr`), knowing
/// nothing of `f`'s parameters: a loop missing ends after a number of passes the count can't
/// bound.
pub fn loop_passes(m: &Module, f: FuncId) -> HashMap<usize, u32> {
    let mut c = Counter {
        m,
        steps: 0,
        gave_up: false,
        bounds: false,
        memo: HashMap::new(),
        breaches: Vec::new(),
        passes: HashMap::new(),
        proof: None,
        totals: None,
    };
    let args = vec![None; m.functions[f.0 as usize].params.len()];
    c.call(f, &args);
    if c.gave_up { HashMap::new() } else { c.passes }
}

/// What the work count proves of the run-time checks a shader is written with (language.md
/// §11.3), so a writer can leave out the ones it makes unneeded. Each holds on every way the
/// count follows, and the count follows every way the code can run (a bound may be loose, never
/// low), with what it knows of the most an integer can be ([`entry_work`]'s count, which the
/// host's step guard trusts). What a loop it can't bound computes is not proved, nor anything
/// once the count gives up.
#[derive(Clone, Debug, Default)]
pub struct Proofs {
    /// The loops, by their bodies' addresses, that end within [`MOST_PASSES`] passes every time
    /// they run: a loop the GPU's compiler can see ends needs no counter to make it.
    pub loops: std::collections::HashSet<usize>,
    /// The most each `u32` value used as an index can be, by its function's address and the
    /// value.
    pub index_most: HashMap<(usize, ValueId), u64>,
    /// The integer divisors, by their function's address and the value, that are a constant
    /// every time they're used, never 0, nor -1 of a signed division.
    pub divisors: std::collections::HashSet<(usize, ValueId)>,
}

impl Proofs {
    /// Whether index `i` in function `f` is always less than `len`.
    pub fn in_range(&self, f: &Function, i: ValueId, len: u32) -> bool {
        let key = (f as *const Function as usize, i);
        self.index_most.get(&key).is_some_and(|m| *m < u64::from(len))
    }

    /// Whether divisor `d` in function `f` is never 0 (nor -1 of a signed division).
    pub fn safe_divisor(&self, f: &Function, d: ValueId) -> bool {
        self.divisors.contains(&(f as *const Function as usize, d))
    }

    /// Whether the loop with body `body` always ends within [`MOST_PASSES`] passes.
    pub fn ends(&self, body: &[Stmt]) -> bool {
        self.loops.contains(&(body.as_ptr() as usize))
    }
}

/// [`Proofs`] of module `m`, followed from each entry point knowing nothing of its inputs.
pub fn proofs(m: &Module) -> Proofs {
    let mut c = Counter {
        m,
        steps: 0,
        gave_up: false,
        bounds: true,
        memo: HashMap::new(),
        breaches: Vec::new(),
        passes: HashMap::new(),
        proof: Some(ProofLog::default()),
        totals: None,
    };
    for e in &m.entry_points {
        c.steps = 0;
        c.call(e.function, &[]);
        if c.gave_up {
            return Proofs::default();
        }
    }
    let log = c.proof.take().expect("the proof log");
    Proofs {
        loops: c.passes.keys().filter(|b| !log.endless.contains(b)).copied().collect(),
        index_most: log.index.into_iter().filter_map(|(k, m)| Some((k, m?))).collect(),
        divisors: log.div.into_iter().filter(|(_, ok)| *ok).map(|(k, _)| k).collect(),
    }
}

/// What a [`proofs`] count has found so far, and the order it found it in, so what a loop it
/// can't bound found can be taken back.
#[derive(Default)]
struct ProofLog {
    /// The most of each index (`None`: not known on some way).
    index: HashMap<(usize, ValueId), Option<u64>>,
    /// Whether each divisor was safe every time.
    div: HashMap<(usize, ValueId), bool>,
    /// What was recorded, in order: an index (`true`) or a divisor.
    log: Vec<(bool, (usize, ValueId))>,
    /// The loops found unbounded, by their bodies' addresses.
    endless: std::collections::HashSet<usize>,
}

impl ProofLog {
    fn index(&mut self, key: (usize, ValueId), most: Option<u64>) {
        let e = self.index.entry(key).or_insert(most);
        *e = match (*e, most) {
            (Some(a), Some(b)) => Some(a.max(b)),
            _ => None,
        };
        self.log.push((true, key));
    }

    fn divisor(&mut self, key: (usize, ValueId), ok: bool) {
        *self.div.entry(key).or_insert(ok) &= ok;
        self.log.push((false, key));
    }

    /// Takes back what was found since `from`: a loop the count couldn't bound computed it.
    fn unprove(&mut self, from: usize, body: usize) {
        for (index, key) in self.log.split_off(from) {
            if index {
                self.index.insert(key, None);
            } else {
                self.div.insert(key, false);
            }
        }
        self.endless.insert(body);
    }
}

#[derive(Clone, Debug)]
struct Called {
    work: u64,
    unbounded: bool,
    loop_at: Option<Span>,
    ret: Option<Const>,
}

/// A call, by its function and what's known of its arguments.
type CallKey = (u32, Known);

/// What's known of a call's arguments: of each, a kind and a value. Only what control can
/// depend on is told apart: integers and `bool`s.
pub type Known = Vec<Option<(u8, u32)>>;

fn known_of(args: &[Option<Const>]) -> Known {
    args.iter()
        .map(|a| match a {
            Some(Const::Bool(b)) => Some((0, u32::from(*b))),
            Some(Const::I32(i)) => Some((1, *i as u32)),
            Some(Const::U32(u)) => Some((2, *u)),
            _ => None,
        })
        .collect()
}

fn consts_of(k: &Known) -> Vec<Option<Const>> {
    k.iter()
        .map(|a| match a {
            Some((0, b)) => Some(Const::Bool(*b != 0)),
            Some((1, i)) => Some(Const::I32(*i as i32)),
            Some((2, u)) => Some(Const::U32(*u)),
            _ => None,
        })
        .collect()
}

/// A call's runs in one call of the function it's in ([`CallRuns`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SiteRuns {
    /// How many times it runs: each of its loops' passes (one, for a loop the count can't
    /// bound), every side of a branch the count doesn't know taken.
    pub times: u64,
    /// All those runs' work, each given what's known of its arguments then.
    pub work: u64,
    /// A run's work is unbounded, or the count gave up before it was sure of them all.
    pub unbounded: bool,
    /// What's known of its arguments, each way it differs between runs, with how many runs (a
    /// way may be left with none).
    pub known: Vec<(Known, u64)>,
}

/// What a [`CallRuns`] count has found so far: each call's runs, and each run's record (the
/// call, which of its `known`, how many runs, their work), so a loop the count can't bound can
/// take back all but a pass of what it added.
#[derive(Default)]
struct SiteLog {
    runs: HashMap<usize, SiteRuns>,
    log: Vec<(usize, u32, u64, u64)>,
    /// The loops found unbounded (by their bodies' addresses): met again, each runs one pass.
    endless: std::collections::HashSet<usize>,
}

impl SiteLog {
    fn record(&mut self, at: usize, known: Known, work: u64, unbounded: bool) {
        let r = self.runs.entry(at).or_default();
        r.times = r.times.saturating_add(1);
        r.work = r.work.saturating_add(work);
        r.unbounded |= unbounded;
        let k = match r.known.iter().position(|(j, _)| *j == known) {
            Some(k) => k,
            None => {
                r.known.push((known, 0));
                r.known.len() - 1
            }
        };
        r.known[k].1 += 1;
        self.log.push((at, k as u32, 1, work));
    }

    /// Scales the runs recorded since `from` down to one of the loop's `passes`: the count can't
    /// bound it. What was known of a call's arguments then is what all of those runs agree on.
    fn one_pass(&mut self, from: usize, passes: u64) {
        let tail = self.log.split_off(from);
        // (Each call: its runs, their work, and how many of them each way of `known` had.)
        type Added = (usize, u64, u64, Vec<(u32, u64)>);
        let mut each: Vec<Added> = Vec::new();
        for (at, k, n, w) in tail {
            let e = match each.iter().position(|e| e.0 == at) {
                Some(i) => &mut each[i],
                None => {
                    each.push((at, 0, 0, Vec::new()));
                    each.last_mut().expect("pushed")
                }
            };
            e.1 += n;
            e.2 = e.2.saturating_add(w);
            match e.3.iter_mut().find(|(j, _)| *j == k) {
                Some((_, m)) => *m += n,
                None => e.3.push((k, n)),
            }
        }
        let passes = passes.max(1);
        for (at, n, w, ks) in each {
            let r = self.runs.get_mut(&at).expect("a recorded call");
            let (n1, w1) = (n.div_ceil(passes), w.div_ceil(passes));
            r.times = r.times - n + n1;
            r.work = r.work - w + w1;
            let mut agreed: Option<Known> = None;
            for (k, m) in ks {
                let (known, count) = &mut r.known[k as usize];
                *count -= m;
                agreed = Some(match agreed {
                    None => known.clone(),
                    Some(a) => a
                        .iter()
                        .zip(known.iter())
                        .map(|(x, y)| if x == y { *x } else { None })
                        .collect(),
                });
            }
            let agreed = agreed.expect("a run");
            let k = match r.known.iter().position(|(j, _)| *j == agreed) {
                Some(k) => k,
                None => {
                    r.known.push((agreed, 0));
                    r.known.len() - 1
                }
            };
            r.known[k].1 += n1;
            self.log.push((at, k as u32, n1, w1));
        }
    }
}

/// The most statements a [`CallRuns`] follows in all (a guard on the count's own time): past
/// it, the calls of what's counted after are unbounded.
pub const MOST_CALL_RUN_STEPS: u64 = 1 << 26;

/// How each call in a function runs in one call of it, knowing what the count knows: its
/// constants, what's known of its arguments, and the values computed from them. One count
/// serves the calls it makes, so a function called many times is followed once for each set of
/// known arguments.
pub struct CallRuns<'a> {
    c: Counter<'a>,
    /// The statements followed so far, over every `of`.
    steps: u64,
}

impl<'a> CallRuns<'a> {
    pub fn new(m: &'a Module) -> CallRuns<'a> {
        CallRuns {
            c: Counter {
                m,
                steps: 0,
                gave_up: false,
                bounds: false,
                memo: HashMap::new(),
                breaches: Vec::new(),
                passes: HashMap::new(),
                proof: None,
                totals: None,
            },
            steps: 0,
        }
    }

    /// The most work one call of `f` can do given `known` of its arguments (`None`: unbounded).
    pub fn work_of(&mut self, f: FuncId, known: &Known) -> Option<u64> {
        (self.c.steps, self.c.gave_up) = (0, false);
        let c = self.c.call(f, &consts_of(known));
        (!c.unbounded && !self.c.gave_up).then_some(c.work)
    }

    /// Each call in function `f`'s own body, by the call expression's address, with its runs
    /// in one call of `f` given `known` of its arguments, and whether the count was sure of them
    /// all. Sure, a call missing doesn't run; not (past [`MOST_STEPS`] in `f`, or
    /// [`MOST_CALL_RUN_STEPS`] in all), each found is unbounded and one missing may run.
    pub fn of(&mut self, f: FuncId, known: &Known) -> (HashMap<usize, SiteRuns>, bool) {
        if self.steps > MOST_CALL_RUN_STEPS {
            return (HashMap::new(), false);
        }
        let func = &self.c.m.functions[f.0 as usize];
        let mut args = consts_of(known);
        args.resize(func.params.len(), None);
        let mut fr = Frame {
            f: func,
            args,
            vals: HashMap::new(),
            most: HashMap::new(),
            parts: HashMap::new(),
            nots: HashMap::new(),
            work: 0,
            unbounded: false,
            loop_at: None,
            here: None,
            ret: None,
            calls: HashMap::new(),
            sites: Some(SiteLog::default()),
            exits: Vec::new(),
            left: 0,
        };
        (self.c.steps, self.c.gave_up) = (0, false);
        self.c.block(&mut fr, &mut State::default(), &func.body);
        self.steps = self.steps.saturating_add(self.c.steps);
        let mut runs = fr.sites.map(|s| s.runs).unwrap_or_default();
        let sure = !self.c.gave_up;
        if !sure {
            for r in runs.values_mut() {
                r.unbounded = true;
            }
        }
        (runs, sure)
    }
}

struct Counter<'a> {
    m: &'a Module,
    /// Whether the count uses what it knows of the most an integer can be (a clamp), and the
    /// parts of a value built of them, and follows a loop to an exit after its last pass: the
    /// work bound a host weighs a dispatch by (`module_work`, `entry_work`). The counts E0712
    /// and `wrela pipelines`' few-input calls take (`loop_passes`, `ExprWork`, `CallRuns`) don't,
    /// so their findings are as they were.
    bounds: bool,
    steps: u64,
    /// The count went past [`MOST_STEPS`]: what it found after is not the code's, so no
    /// function is reported over its budget from then on.
    gave_up: bool,
    memo: HashMap<CallKey, Called>,
    breaches: Vec<Breach>,
    /// The most passes each loop that ended made, by its body's address.
    passes: HashMap<usize, u32>,
    /// What the count proves of the run-time checks ([`proofs`]), when it's asked.
    proof: Option<ProofLog>,
    /// Each loop's most work in all its passes, by its body's address, `None` once a way
    /// through it had no bound ([`loop_totals`]), when it's asked.
    totals: Option<HashMap<usize, Option<u64>>>,
}

/// How a block ended: `Normal` when some way through it goes on past it (the state is what's
/// known on those ways), otherwise how every way left it (`Leaves`: not all the same way). A
/// way that leaves by `break` or `continue` leaves its state with its loop ([`Exits`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Flow {
    Normal,
    Break,
    Continue,
    Return,
    Leaves,
}

#[derive(Clone, Debug, Default)]
struct State {
    /// Locals whose whole value is known.
    locals: HashMap<LocalId, Const>,
    /// Integer locals whose value isn't known, with the most each can be.
    most: HashMap<LocalId, i64>,
    /// Locals that hold a value built of parts (a constant table), by that value.
    built: HashMap<LocalId, ValueId>,
    /// `bool` locals whose whole value is a value not known, by that value: on each side of an
    /// `if` on it (or on its negation), they are known. `a && b` is `r = a; if a { r = b }`,
    /// so where `b` is known false, `r` is false on both sides.
    copies: HashMap<LocalId, ValueId>,
}

impl State {
    /// The most integer local `l` can be here: its value, or what's known of its most.
    fn bound(&self, l: &LocalId) -> Option<i64> {
        self.locals.get(l).and_then(int_of).or_else(|| self.most.get(l).copied())
    }

    /// What's known on either of two ways: a value both agree on, and of an integer the most
    /// either way can make it (a value one way and another the other is at most the larger).
    fn join(&mut self, other: &State) {
        let mut most = HashMap::new();
        for (l, c) in &self.locals {
            if other.locals.get(l) != Some(c)
                && let (Some(a), Some(b)) = (int_of(c), other.bound(l))
            {
                most.insert(*l, a.max(b));
            }
        }
        for (l, m) in &self.most {
            if let Some(b) = other.bound(l) {
                most.insert(*l, (*m).max(b));
            }
        }
        self.locals.retain(|l, c| other.locals.get(l) == Some(c));
        self.built.retain(|l, v| other.built.get(l) == Some(v));
        self.copies.retain(|l, v| other.copies.get(l) == Some(v));
        self.most = most;
    }

    /// Forgets what a loop the count couldn't follow to its end may have changed since `entry`
    /// (what it knew when the loop began): only what's the same as then is still sure.
    fn keep_unchanged(&mut self, entry: &State) {
        self.locals.retain(|l, c| entry.locals.get(l) == Some(c));
        self.most.retain(|l, m| entry.most.get(l) == Some(m));
        self.built.retain(|l, v| entry.built.get(l) == Some(v));
        self.copies.retain(|l, v| entry.copies.get(l) == Some(v));
    }

    /// What's known on the side of an `if` on `cond` where it is `holds`: each `bool` local
    /// that is `cond` (or its negation, `nots`) is that.
    fn on_side(&mut self, cond: ValueId, holds: bool, nots: &HashMap<ValueId, ValueId>) {
        let neg = nots.get(&cond).copied();
        let mut known = Vec::new();
        for (l, v) in &self.copies {
            if *v == cond {
                known.push((*l, holds));
            } else if Some(*v) == neg {
                known.push((*l, !holds));
            }
        }
        for (l, b) in known {
            self.locals.insert(l, Const::Bool(b));
            self.copies.remove(&l);
        }
    }
}

/// The ways out of a loop the count is following: what's known where each `break` left it
/// (all passes), and where each `continue` left the pass being followed.
#[derive(Default)]
struct Exits {
    breaks: Option<State>,
    continues: Option<State>,
}

fn join_into(slot: &mut Option<State>, st: &State) {
    match slot {
        Some(s) => s.join(st),
        None => *slot = Some(st.clone()),
    }
}

struct Frame<'f> {
    f: &'f Function,
    args: Vec<Option<Const>>,
    vals: HashMap<ValueId, Const>,
    /// Integer values not known, with the most each can be ([`most_of`]).
    most: HashMap<ValueId, i64>,
    /// Values built of parts (a constant table's elements, a vector's components): an element
    /// read at a known place is its part.
    parts: HashMap<ValueId, Vec<ValueId>>,
    /// Values that are the negation of another (`!x`), by `x`.
    nots: HashMap<ValueId, ValueId>,
    work: u64,
    unbounded: bool,
    loop_at: Option<Span>,
    here: Option<Span>,
    /// What each sure return gave: `Some(None)` once two differ or one isn't known.
    ret: Option<Option<Const>>,
    /// Work of each function it calls itself.
    calls: HashMap<u32, u64>,
    /// Each call in its own body, by the call's address, and its runs ([`CallRuns`] only).
    sites: Option<SiteLog>,
    /// The loops being followed, innermost last, and the ways out of each found so far.
    exits: Vec<Exits>,
    /// The most work any way through the function that has already returned did: a side of an
    /// `if` that returns ends its way there, so the code after the `if` is counted only on the
    /// sides that go on, and the call's work is the most of every way.
    left: u64,
}

impl<'a> Counter<'a> {
    fn call(&mut self, id: FuncId, args: &[Option<Const>]) -> Called {
        let key = (id.0, known_of(args));
        let args = consts_of(&key.1);
        if let Some(c) = self.memo.get(&key) {
            return c.clone();
        }
        let f = &self.m.functions[id.0 as usize];
        let mut fr = Frame {
            f,
            args,
            vals: HashMap::new(),
            most: HashMap::new(),
            parts: HashMap::new(),
            nots: HashMap::new(),
            work: 0,
            unbounded: false,
            loop_at: None,
            here: None,
            ret: None,
            calls: HashMap::new(),
            sites: None,
            exits: Vec::new(),
            left: 0,
        };
        let mut st = State::default();
        self.block(&mut fr, &mut st, &f.body);
        let called = Called {
            work: fr.work.max(fr.left),
            unbounded: fr.unbounded,
            loop_at: fr.loop_at,
            ret: fr.ret.flatten(),
        };
        if let Some(Budget { work: Some(budget), at, .. }) = f.budget
            && !self.gave_up
        {
            let over = called.unbounded || called.work > budget;
            if over {
                let mut biggest: Vec<(String, u64)> = fr
                    .calls
                    .into_iter()
                    .map(|(c, w)| (self.m.functions[c as usize].name.clone(), w))
                    .collect();
                biggest.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                biggest.truncate(3);
                let b = Breach {
                    function: f.name.clone(),
                    at,
                    budget,
                    work: (!called.unbounded).then_some(called.work),
                    unbounded: called.loop_at,
                    biggest,
                };
                match self.breaches.iter_mut().find(|x| x.at == at) {
                    Some(x) => {
                        if b.work.is_none() || x.work.is_some_and(|w| b.work.is_some_and(|v| v > w))
                        {
                            *x = b;
                        }
                    }
                    None => self.breaches.push(b),
                }
            }
        }
        if !self.gave_up {
            self.memo.insert(key, called.clone());
        }
        called
    }

    fn block(&mut self, fr: &mut Frame, st: &mut State, b: &[Stmt]) -> Flow {
        for s in b {
            self.steps += 1;
            if self.steps > MOST_STEPS {
                self.gave_up = true;
                fr.unbounded = true;
                return Flow::Return;
            }
            match self.stmt(fr, st, s) {
                Flow::Normal => {}
                other => return other,
            }
        }
        Flow::Normal
    }

    fn stmt(&mut self, fr: &mut Frame, st: &mut State, s: &Stmt) -> Flow {
        match s {
            Stmt::At(span) => fr.here = Some(*span),
            Stmt::Let(v, e) => {
                // A value a loop's pass defines again forgets what an earlier pass gave it.
                // (A local that held its earlier pass's value holds no parts of this one's.)
                st.built.retain(|_, x| x != v);
                st.copies.retain(|_, x| x != v);
                match e {
                    Expr::Construct(_, parts) if self.bounds => fr.parts.insert(*v, parts.clone()),
                    _ => fr.parts.remove(v),
                };
                match e {
                    Expr::Unary(UnOp::Not, x) => fr.nots.insert(*v, *x),
                    _ => fr.nots.remove(v),
                };
                match self.expr(fr, st, e) {
                    Some(c) => {
                        fr.most.remove(v);
                        fr.vals.insert(*v, c)
                    }
                    None => {
                        let most =
                            if self.bounds { most_of(&self.m.types, fr, st, *v, e) } else { None };
                        match most {
                            Some(m) => fr.most.insert(*v, m),
                            None => fr.most.remove(v),
                        };
                        fr.vals.remove(v)
                    }
                };
            }
            Stmt::Eval(e) => {
                self.expr(fr, st, e);
            }
            Stmt::Store(p, v) => {
                self.prove_place(fr, p);
                fr.work = fr.work.saturating_add(self.place_cost(p, true));
                match (p.root.clone(), p.path.is_empty()) {
                    (PlaceRoot::Local(l), true) => {
                        match fr.vals.get(v) {
                            Some(c) => {
                                st.locals.insert(l, c.clone());
                                st.copies.remove(&l);
                            }
                            None => {
                                st.locals.remove(&l);
                                let boolean = matches!(
                                    self.m.types.get(fr.f.value_ty(*v)),
                                    TypeDef::Scalar(Scalar::Bool)
                                );
                                match boolean {
                                    true => st.copies.insert(l, *v),
                                    false => st.copies.remove(&l),
                                };
                            }
                        }
                        match fr.most.get(v) {
                            Some(m) => st.most.insert(l, *m),
                            None => st.most.remove(&l),
                        };
                        match fr.parts.contains_key(v) {
                            true => st.built.insert(l, *v),
                            false => st.built.remove(&l),
                        };
                    }
                    (PlaceRoot::Local(l), false) => {
                        st.locals.remove(&l);
                        st.most.remove(&l);
                        st.built.remove(&l);
                        st.copies.remove(&l);
                    }
                    _ => {}
                }
            }
            Stmt::If { cond, then, else_ } => {
                fr.work = fr.work.saturating_add(OP);
                match fr.vals.get(cond) {
                    Some(Const::Bool(true)) => return self.block(fr, st, then),
                    Some(Const::Bool(false)) => return self.block(fr, st, else_),
                    _ => return self.both(fr, st, *cond, then, else_),
                }
            }
            Stmt::Loop { body, continuing } => return self.lp(fr, st, body, continuing),
            Stmt::Break => {
                if let Some(x) = fr.exits.last_mut() {
                    join_into(&mut x.breaks, st);
                }
                return Flow::Break;
            }
            Stmt::Continue => {
                if let Some(x) = fr.exits.last_mut() {
                    join_into(&mut x.continues, st);
                }
                return Flow::Continue;
            }
            Stmt::Return(v) => {
                let c = v.and_then(|v| fr.vals.get(&v).cloned());
                fr.ret = Some(match fr.ret.take() {
                    None => c,
                    Some(prev) => prev.filter(|p| Some(p) == c.as_ref()),
                });
                return Flow::Return;
            }
            Stmt::Trap => return Flow::Return,
        }
        Flow::Normal
    }

    /// An `if` on a condition the count doesn't know: the costlier side's work, and what's
    /// known after it on the sides that go on. A side that leaves by `break` or `continue` left
    /// what it knew with its loop, which joins it with the loop's other ways out.
    fn both(
        &mut self,
        fr: &mut Frame,
        st: &mut State,
        cond: ValueId,
        then: &[Stmt],
        else_: &[Stmt],
    ) -> Flow {
        let start = fr.work;
        let mut a = st.clone();
        a.on_side(cond, true, &fr.nots);
        let fa = self.block(fr, &mut a, then);
        let wa = fr.work - start;
        fr.work = start;
        let mut b = st.clone();
        b.on_side(cond, false, &fr.nots);
        let fb = self.block(fr, &mut b, else_);
        let wb = fr.work - start;
        fr.work = start.saturating_add(wa.max(wb));
        match (fa, fb) {
            // One side returns: its way ends with its own work, and only the other's goes on.
            (Flow::Normal, Flow::Return) => {
                fr.left = fr.left.max(start.saturating_add(wb));
                fr.work = start.saturating_add(wa);
                *st = a;
                Flow::Normal
            }
            (Flow::Return, Flow::Normal) => {
                fr.left = fr.left.max(start.saturating_add(wa));
                fr.work = start.saturating_add(wb);
                *st = b;
                Flow::Normal
            }
            (x, y) if x == y => {
                a.join(&b);
                *st = a;
                x
            }
            (Flow::Normal, _) => {
                *st = a;
                Flow::Normal
            }
            (_, Flow::Normal) => {
                *st = b;
                Flow::Normal
            }
            // Each side leaves, differently: what follows doesn't run.
            _ => {
                a.join(&b);
                *st = a;
                Flow::Leaves
            }
        }
    }

    fn lp(&mut self, fr: &mut Frame, st: &mut State, body: &[Stmt], continuing: &[Stmt]) -> Flow {
        let at = fr.here;
        let mut passes = 0u32;
        // (Counting its calls' runs, a loop goes on past an unbounded call in it: its passes
        // are still sure. Unbounded itself, each call in it counts one pass, and met again, it
        // runs one.)
        let mark = fr.sites.as_ref().map(|s| s.log.len());
        let body_at = body.as_ptr() as usize;
        let endless = fr.sites.as_ref().is_some_and(|s| s.endless.contains(&body_at));
        let proof_mark = self.proof.as_ref().map(|p| p.log.len());
        let mut ran = 0u64;
        let work_at = fr.work;
        // (Each pass is followed knowing what's sure on every way into it, so its work is the
        // most any of them does; the loop ends when no way goes on to another pass, and what's
        // known after it is what every `break` that left it knew, from every pass.)
        let entry = st.clone();
        fr.exits.push(Exits::default());
        loop {
            // (Its exit is looked for once more after its last pass: a loop of `MOST_PASSES`
            // passes is bounded.)
            let most = if self.bounds { MOST_PASSES + 1 } else { MOST_PASSES };
            let done = if mark.is_some() {
                passes >= most || (endless && passes >= 1)
            } else {
                passes >= most || fr.unbounded
            };
            if done {
                fr.unbounded = true;
                if let Some(t) = self.totals.as_mut() {
                    t.insert(body_at, None);
                }
                if let (Some(p), Some(mark)) = (self.proof.as_mut(), proof_mark) {
                    p.unprove(mark, body_at);
                }
                if fr.loop_at.is_none() {
                    fr.loop_at = at;
                }
                if let (Some(sites), Some(mark)) = (fr.sites.as_mut(), mark)
                    && !endless
                {
                    sites.one_pass(mark, ran);
                    sites.endless.insert(body_at);
                }
                // Not followed to its end: only what no pass followed changed is still known.
                let exits = fr.exits.pop().expect("this loop's exits");
                if let Some(b) = &exits.breaks {
                    st.join(b);
                }
                st.keep_unchanged(&entry);
                return Flow::Normal;
            }
            passes += 1;
            ran += 1;
            let before = st.locals.clone();
            let flow = self.block(fr, st, body);
            let x = fr.exits.last_mut().expect("this loop's exits");
            let continues = x.continues.take();
            // What's known on every way on to `continuing`: through the body's end, and from
            // each `continue`.
            let on = match (flow, continues) {
                (Flow::Normal, Some(c)) => {
                    st.join(&c);
                    true
                }
                (Flow::Normal, None) => true,
                (_, Some(c)) => {
                    *st = c;
                    true
                }
                (_, None) => false,
            };
            let on = on && self.block(fr, st, continuing) == Flow::Normal;
            if !on {
                // No way goes on to another pass: after the loop is what each `break` knew.
                let exits = fr.exits.pop().expect("this loop's exits");
                if let Some(t) = self.totals.as_mut() {
                    let total = fr.work.saturating_sub(work_at);
                    let e = t.entry(body_at).or_insert(Some(0));
                    if let Some(most) = e {
                        *most = (*most).max(total);
                    }
                }
                return match exits.breaks {
                    Some(b) => {
                        *st = b;
                        let most = self.passes.entry(body.as_ptr() as usize).or_default();
                        *most = (*most).max(passes);
                        Flow::Normal
                    }
                    None => Flow::Return,
                };
            }
            // A pass that ends where it began runs the same way forever: no exit is sure.
            if st.locals == before {
                passes = MOST_PASSES + 1;
            }
        }
    }

    /// Counts `e`'s work, and gives its value when the count knows it.
    fn expr(&mut self, fr: &mut Frame, st: &mut State, e: &Expr) -> Option<Const> {
        let m = self.m;
        let comps = |v: &ValueId| comps(&m.types, fr.f.value_ty(*v));
        let known = |v: &ValueId| fr.vals.get(v).cloned();
        let (w, value): (u64, Option<Const>) = match e {
            Expr::Const(c) => (0, Some(c.clone())),
            Expr::Extract(v, i) => (0, self.part(fr, *v, Some(u64::from(*i)))),
            Expr::ExtractDyn(v, i) => {
                self.prove_index(fr, *i);
                let at = known(i).and_then(|c| int_of(&c)).and_then(|i| u64::try_from(i).ok());
                (OP, at.and_then(|at| self.part(fr, *v, Some(at))))
            }
            Expr::Zero(_)
            | Expr::Splat(..)
            | Expr::Swizzle(..)
            | Expr::Construct(..)
            | Expr::Variant(..)
            | Expr::Bitcast(..)
            | Expr::Run(_)
            | Expr::Addr(_)
            | Expr::EntryInput(_)
            | Expr::Discard
            | Expr::Host(..)
            | Expr::Mem(..) => (0, None),
            Expr::Param(i) => (0, fr.args.get(*i as usize).cloned().flatten()),
            Expr::Load(p) => {
                self.prove_place(fr, p);
                let v = match (&p.root, p.path.as_slice()) {
                    (PlaceRoot::Local(l), []) => st.locals.get(l).cloned(),
                    // An element of a table a local holds, at a known place.
                    (PlaceRoot::Local(l), [Proj::Index(i)]) => {
                        let at =
                            known(i).and_then(|c| int_of(&c)).and_then(|i| u64::try_from(i).ok());
                        st.built.get(l).and_then(|v| self.part(fr, *v, at))
                    }
                    _ => None,
                };
                (self.place_cost(p, false), v)
            }
            Expr::Unary(op, x) => {
                let v = match (op, known(x)) {
                    (UnOp::Not, Some(Const::Bool(b))) => Some(Const::Bool(!b)),
                    (UnOp::Neg, Some(Const::I32(i))) => Some(Const::I32(i.wrapping_neg())),
                    _ => None,
                };
                (OP * comps(x), v)
            }
            Expr::Binary(op, a, b) => {
                if matches!(op, BinOp::Div | BinOp::Rem) {
                    self.prove_divisor(fr, *b);
                }
                let n = binary_cost(&m.types, fr.f, *op, *a, *b);
                let v = match (known(a), known(b)) {
                    (Some(x), Some(y)) => fold(*op, x, y),
                    // One side settles it: `done && i < n` is false once `i` reaches `n`,
                    // whatever `done` is (a loop's exit the count can see).
                    (Some(Const::Bool(false)), _) | (_, Some(Const::Bool(false)))
                        if *op == BinOp::And =>
                    {
                        Some(Const::Bool(false))
                    }
                    (Some(Const::Bool(true)), _) | (_, Some(Const::Bool(true)))
                        if *op == BinOp::Or =>
                    {
                        Some(Const::Bool(true))
                    }
                    (x, y) => compare_most(
                        *op,
                        (x.as_ref().and_then(int_of), fr.most.get(a).copied()),
                        (y.as_ref().and_then(int_of), fr.most.get(b).copied()),
                    ),
                };
                (n, v)
            }
            Expr::Builtin(b, args) => (builtin_cost(*b, args.first().map_or(1, comps)), None),
            Expr::ArrayLength(_) => (OP, None),
            Expr::Convert(x, _) => (OP * comps(x), None),
            Expr::Select { cond, if_true, if_false } => {
                let v = match known(cond) {
                    Some(Const::Bool(true)) => known(if_true),
                    Some(Const::Bool(false)) => known(if_false),
                    _ => None,
                };
                (OP * comps(if_true), v)
            }
            Expr::Texture(op, ..) => (texture_cost(*op), None),
            Expr::TextureStore(..) => (TEXEL, None),
            Expr::Atomic(_, p, _) => {
                self.prove_place(fr, p);
                (ATOMIC, None)
            }
            Expr::Subgroup(..) => (2 * OP, None),
            Expr::Barrier => (4 * OP, None),
            Expr::Call(id, args) => {
                let vals: Vec<Option<Const>> = args
                    .iter()
                    .map(|a| match a {
                        Arg::Value(v) => known(v),
                        Arg::Place(_) => None,
                    })
                    .collect();
                for a in args {
                    if let Arg::Place(p) = a {
                        self.prove_place(fr, p);
                    }
                }
                // A place lent to the call may be written there.
                for a in args {
                    if let Arg::Place(p) = a
                        && let PlaceRoot::Local(l) = p.root
                    {
                        st.locals.remove(&l);
                        st.most.remove(&l);
                        st.built.remove(&l);
                    }
                }
                let c = self.call(*id, &vals);
                if c.unbounded {
                    fr.unbounded = true;
                    if fr.loop_at.is_none() {
                        fr.loop_at = c.loop_at;
                    }
                }
                if let Some(sites) = fr.sites.as_mut() {
                    sites.record(e as *const Expr as usize, known_of(&vals), c.work, c.unbounded);
                }
                let e = fr.calls.entry(id.0).or_insert(0);
                *e = e.saturating_add(c.work);
                (c.work, c.ret)
            }
        };
        fr.work = fr.work.saturating_add(w);
        value
    }

    /// Records the most each index of place `p` can be here ([`proofs`]).
    fn prove_place(&mut self, fr: &Frame, p: &Place) {
        if self.proof.is_some() {
            for x in &p.path {
                if let Proj::Index(i) = x {
                    self.prove_index(fr, *i);
                }
            }
        }
    }

    /// Records the most index `i` can be here: its value, or of a `u32`, the most it can be.
    fn prove_index(&mut self, fr: &Frame, i: ValueId) {
        let Some(log) = self.proof.as_mut() else { return };
        let most = match fr.vals.get(&i).and_then(int_of) {
            Some(k) => u64::try_from(k).ok(),
            None if matches!(self.m.types.get(fr.f.value_ty(i)), TypeDef::Scalar(Scalar::U32)) => {
                fr.most.get(&i).and_then(|m| u64::try_from(*m).ok())
            }
            None => None,
        };
        log.index((fr.f as *const Function as usize, i), most);
    }

    /// Records whether integer divisor `d` is safe here: a constant, not 0, and not -1 of a
    /// signed division.
    fn prove_divisor(&mut self, fr: &Frame, d: ValueId) {
        let Some(log) = self.proof.as_mut() else { return };
        let ok = match fr.vals.get(&d) {
            Some(Const::U32(k)) => *k != 0,
            Some(Const::I32(k)) => *k != 0 && *k != -1,
            _ => false,
        };
        log.divisor((fr.f as *const Function as usize, d), ok);
    }

    /// The known value of part `at` of value `v`, built of parts.
    fn part(&self, fr: &Frame, v: ValueId, at: Option<u64>) -> Option<Const> {
        let parts = fr.parts.get(&v)?;
        let p = parts.get(usize::try_from(at?).ok()?)?;
        fr.vals.get(p).cloned()
    }

    /// A read (or a write) of `p`: free for a local or a parameter, which are registers.
    fn place_cost(&self, p: &Place, write: bool) -> u64 {
        let index = p.path.iter().any(|x| matches!(x, Proj::Index(_)));
        let base = match &p.root {
            PlaceRoot::Resource(r) => match self.m.resources[r.0 as usize].kind {
                ResourceKind::Uniform { .. } if !write => OP,
                ResourceKind::Workgroup => SHARED,
                ResourceKind::Private => 0,
                _ => MEMORY,
            },
            _ => 0,
        };
        base + if index { OP } else { 0 }
    }
}

/// How many components a value of type `t` has: what elementwise arithmetic on it costs.
fn comps(types: &Types, t: TypeId) -> u64 {
    match types.get(t) {
        TypeDef::Vector(_, n) => u64::from(*n),
        TypeDef::Matrix(c, r) => u64::from(*c) * u64::from(*r),
        _ => 1,
    }
}

fn binary_cost(types: &Types, f: &Function, op: BinOp, a: ValueId, b: ValueId) -> u64 {
    let (ta, tb) = (types.get(f.value_ty(a)), types.get(f.value_ty(b)));
    let per = if matches!(op, BinOp::Div | BinOp::Rem) { SLOW_OP } else { OP };
    let n = match (ta, tb) {
        // A matrix product: a multiply-add for each element of the result, each a row's length.
        (TypeDef::Matrix(c, r), TypeDef::Vector(..)) if op == BinOp::Mul => {
            u64::from(*c) * u64::from(*r)
        }
        (TypeDef::Vector(..), TypeDef::Matrix(c, r)) if op == BinOp::Mul => {
            u64::from(*c) * u64::from(*r)
        }
        (TypeDef::Matrix(k, r), TypeDef::Matrix(c, _)) if op == BinOp::Mul => {
            u64::from(*k) * u64::from(*r) * u64::from(*c)
        }
        _ => comps(types, f.value_ty(a)).max(comps(types, f.value_ty(b))),
    };
    per * n
}

/// A built-in's work, given its first argument's components.
pub fn builtin_cost(b: Builtin, n: u64) -> u64 {
    use Builtin as B;
    match b {
        B::Sqrt
        | B::InverseSqrt
        | B::Sin
        | B::Cos
        | B::Tan
        | B::Asin
        | B::Acos
        | B::Atan
        | B::Atan2
        | B::Exp
        | B::Exp2
        | B::Log
        | B::Log2
        | B::Sinh
        | B::Cosh
        | B::Tanh
        | B::Smoothstep => SLOW_OP * n,
        B::Pow => 2 * SLOW_OP * n,
        B::Clamp | B::Mix | B::Dpdx | B::Dpdy | B::Fwidth => 2 * OP * n,
        B::Dot | B::AllEqual => OP * n,
        B::Length | B::Distance => OP * n + SLOW_OP,
        B::Normalize => 2 * OP * n + SLOW_OP,
        B::Cross => 6 * OP,
        _ => OP * n,
    }
}

fn texture_cost(op: TextureOp) -> u64 {
    match op {
        TextureOp::Load => TEXEL,
        TextureOp::Width
        | TextureOp::Height
        | TextureOp::Depth
        | TextureOp::Layers
        | TextureOp::Levels => OP,
        _ => SAMPLE,
    }
}

/// An integer or `bool` operation on known values, as the GPU computes it.
fn fold(op: BinOp, a: Const, b: Const) -> Option<Const> {
    use std::cmp::Ordering;
    let cmp = |o: Ordering| {
        Some(Const::Bool(match op {
            BinOp::Eq => o == Ordering::Equal,
            BinOp::Ne => o != Ordering::Equal,
            BinOp::Lt => o == Ordering::Less,
            BinOp::Le => o != Ordering::Greater,
            BinOp::Gt => o == Ordering::Greater,
            BinOp::Ge => o != Ordering::Less,
            _ => return None,
        }))
    };
    match (a, b) {
        (Const::I32(x), Const::I32(y)) => match op {
            BinOp::Add | BinOp::WrappingAdd => Some(Const::I32(x.wrapping_add(y))),
            BinOp::Sub | BinOp::WrappingSub => Some(Const::I32(x.wrapping_sub(y))),
            BinOp::Mul | BinOp::WrappingMul => Some(Const::I32(x.wrapping_mul(y))),
            BinOp::Div if y != 0 => Some(Const::I32(x.wrapping_div(y))),
            BinOp::Rem if y != 0 => Some(Const::I32(x.wrapping_rem(y))),
            _ => cmp(x.cmp(&y)),
        },
        (Const::U32(x), Const::U32(y)) => match op {
            BinOp::Add | BinOp::WrappingAdd => Some(Const::U32(x.wrapping_add(y))),
            BinOp::Sub | BinOp::WrappingSub => Some(Const::U32(x.wrapping_sub(y))),
            BinOp::Mul | BinOp::WrappingMul => Some(Const::U32(x.wrapping_mul(y))),
            BinOp::Div if y != 0 => Some(Const::U32(x / y)),
            BinOp::Rem if y != 0 => Some(Const::U32(x % y)),
            BinOp::BitAnd => Some(Const::U32(x & y)),
            BinOp::BitOr => Some(Const::U32(x | y)),
            BinOp::Shl => Some(Const::U32(x.wrapping_shl(y))),
            BinOp::Shr => Some(Const::U32(x.wrapping_shr(y))),
            _ => cmp(x.cmp(&y)),
        },
        (Const::Bool(x), Const::Bool(y)) => match op {
            BinOp::And => Some(Const::Bool(x && y)),
            BinOp::Or => Some(Const::Bool(x || y)),
            BinOp::Eq => Some(Const::Bool(x == y)),
            BinOp::Ne => Some(Const::Bool(x != y)),
            _ => None,
        },
        _ => None,
    }
}

/// A 32-bit integer constant's value.
fn int_of(c: &Const) -> Option<i64> {
    match c {
        Const::I32(i) => Some(i64::from(*i)),
        Const::U32(u) => Some(i64::from(*u)),
        _ => None,
    }
}

/// The most value `v`, defined by `e` and not known itself, can be: for a 32-bit integer made by
/// a clamp, a mask, a remainder or arithmetic on bounded values that can't wrap. Signed values
/// keep only what a clamp or a choice gives (their arithmetic may wrap at either end).
fn most_of(types: &Types, fr: &Frame, st: &State, v: ValueId, e: &Expr) -> Option<i64> {
    let unsigned = match types.get(fr.f.value_ty(v)) {
        TypeDef::Scalar(Scalar::U32) => true,
        TypeDef::Scalar(Scalar::I32) => false,
        _ => return None,
    };
    // The most an operand can be: its value if it's known, else what's known of its most.
    let m = |x: &ValueId| fr.vals.get(x).and_then(int_of).or_else(|| fr.most.get(x).copied());
    let least = |a: Option<i64>, b: Option<i64>| match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    let fits = |x: i64| (x <= i64::from(u32::MAX)).then_some(x);
    match e {
        Expr::Load(p) => match (&p.root, p.path.is_empty()) {
            (PlaceRoot::Local(l), true) => st.most.get(l).copied(),
            _ => None,
        },
        Expr::Builtin(Builtin::Min, args) if args.len() == 2 => least(m(&args[0]), m(&args[1])),
        Expr::Builtin(Builtin::Max, args) if args.len() == 2 => {
            Some(m(&args[0])?.max(m(&args[1])?))
        }
        // `clamp(x, lo, hi)` is `min(max(x, lo), hi)`: never past `hi`, nor past both `x` and
        // `lo` (with `x` under `lo` it is `lo`).
        Expr::Builtin(Builtin::Clamp, args) if args.len() == 3 => {
            least(m(&args[2]), m(&args[0]).zip(m(&args[1])).map(|(x, lo)| x.max(lo)))
        }
        Expr::Select { if_true, if_false, .. } => Some(m(if_true)?.max(m(if_false)?)),
        Expr::Extract(x, i) => fr.parts.get(x).and_then(|p| p.get(*i as usize)).and_then(m),
        Expr::Binary(op, a, b) if unsigned => {
            let (ma, mb) = (m(a), m(b));
            let kb = fr.vals.get(b).and_then(int_of);
            match op {
                BinOp::BitAnd => least(ma, mb),
                // `x % 0` and `x / 0` are `x` or 0 on the GPU: never more than `x`.
                BinOp::Rem => match kb {
                    Some(k) if k > 0 => least(ma, Some(k - 1)),
                    _ => ma,
                },
                BinOp::Div => match kb {
                    Some(k) if k > 0 => ma.map(|x| x / k),
                    _ => ma,
                },
                BinOp::Shr => match kb {
                    Some(k) if (0..32).contains(&k) => ma.map(|x| x >> k),
                    _ => ma,
                },
                BinOp::Add | BinOp::WrappingAdd => fits(ma? + mb?),
                BinOp::Mul | BinOp::WrappingMul => fits(ma?.checked_mul(mb?)?),
                _ => None,
            }
        }
        _ => None,
    }
}

/// A comparison the count can decide from one side's value and the other's most, each side as
/// (its value if known, its most if known): `i < n` is false once `i` reaches the most `n` can
/// be.
fn compare_most(
    op: BinOp,
    a: (Option<i64>, Option<i64>),
    b: (Option<i64>, Option<i64>),
) -> Option<Const> {
    // `x < y` is surely false when x is known and y can't reach it; surely true when x can't
    // reach a known y.
    let lt = |x: (Option<i64>, Option<i64>), y: (Option<i64>, Option<i64>)| match (x, y) {
        ((Some(k), _), (None, Some(most))) if k >= most => Some(false),
        ((None, Some(most)), (Some(k), _)) if most < k => Some(true),
        _ => None,
    };
    let le = |x: (Option<i64>, Option<i64>), y: (Option<i64>, Option<i64>)| match (x, y) {
        ((Some(k), _), (None, Some(most))) if k > most => Some(false),
        ((None, Some(most)), (Some(k), _)) if most <= k => Some(true),
        _ => None,
    };
    let r = match op {
        BinOp::Lt => lt(a, b),
        BinOp::Le => le(a, b),
        BinOp::Gt => lt(b, a),
        BinOp::Ge => le(b, a),
        _ => None,
    };
    r.map(Const::Bool)
}

/// How many statements function `f` comes to with every call written out in place: what a
/// GPU's shader compiler works through, since it copies a function into each place it's called
/// (§18.11, a pipeline's weight). A big function called from many places weighs each time.
pub fn weight(m: &Module, f: FuncId) -> u64 {
    fn go(m: &Module, f: FuncId, memo: &mut HashMap<u32, u64>) -> u64 {
        if let Some(&w) = memo.get(&f.0) {
            return w;
        }
        // A recursion (which GPU code can't have) counts once.
        memo.insert(f.0, 1);
        let mut own = 0u64;
        let mut calls = Vec::new();
        visit::walk(&m.functions[f.0 as usize].body, &mut |s| {
            if !matches!(s, Stmt::At(_)) {
                own += 1;
            }
            if let Some(Expr::Call(id, _)) = s.expr() {
                calls.push(*id);
            }
        });
        let w = calls.into_iter().fold(own, |w, c| w.saturating_add(go(m, c, memo)));
        memo.insert(f.0, w);
        w
    }
    go(m, f, &mut HashMap::new())
}
