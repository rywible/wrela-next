//! The rules. Every rule is legal under the GPU float semantics wrela promises (language.md
//! §11.3 as settled at 60378e5: Metal `relaxed`, infinities and NaNs kept): none assumes a value
//! is finite or not a NaN unless the class's interval says so. Four groups:
//!
//! - **Exact**: the same bits for every input but a zero's sign and a NaN's payload (a GPU's
//!   fast math doesn't keep either).
//! - **Interval**: exact, given a fact the e-class analysis proves (`abs(x) = x` where `x` is
//!   never negative).
//! - **Approx**: within a few ulps of the same value (a reciprocal, an inverse square root, a
//!   logarithm split); a GPU's own fast math rounds as loosely.
//! - **Reassoc**: reassociating and factoring float arithmetic: the rounding of a sum or product
//!   changes, so a result can move by ulps of its largest term. Kept apart and flagged.
//!
//! A constant written `#c` is expanded for each width: `c`, `(splat2 c)`, `(splat3 c)`,
//! `(splat4 c)`. Every application is type-checked: the right side's type, from the matched
//! classes' types, must be the matched class's, else the rule doesn't apply.

use crate::analysis::{Data, Graph, Rate, shape, type_of};
use crate::lang::Node;
use egg::{Applier, ENodeOrVar, Id, Pattern, PatternAst, Rewrite, Subst, Symbol, Var};
use wrela_ir::{Scalar, TypeId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Group {
    Exact,
    Interval,
    Approx,
    Reassoc,
}

/// A fact a rule needs, of its variables' classes.
#[derive(Clone, Copy, Debug)]
pub enum Cond {
    /// Never negative.
    NonNeg(&'static str),
    NonPos(&'static str),
    /// Always more than zero.
    Pos(&'static str),
    /// Finite and never a NaN.
    Clean(&'static str),
    /// Never a NaN, and the first at least the second.
    Ge(&'static str, &'static str),
    /// Never a NaN, and the first less than the second.
    Lt(&'static str, &'static str),
    /// Never a NaN, and the first at most the second.
    Le(&'static str, &'static str),
    /// Never a NaN, and the first within the second and third.
    In(&'static str, &'static str, &'static str),
    /// Never a NaN, and within [0, 1].
    Unit(&'static str),
    /// Within [-m, m]: its square doesn't overflow.
    Bounded(&'static str),
    /// The first changes less often than the second (a uniform divisor of a varying value).
    Slower(&'static str, &'static str),
}

/// The most a [`Cond::Bounded`] value may be: its square is well inside an `f32`.
pub const BOUND: f64 = 1e18;

pub struct Spec {
    pub name: &'static str,
    pub group: Group,
    pub lhs: &'static str,
    pub rhs: &'static str,
    pub conds: &'static [Cond],
}

const fn r(
    name: &'static str,
    group: Group,
    lhs: &'static str,
    rhs: &'static str,
    conds: &'static [Cond],
) -> Spec {
    Spec { name, group, lhs, rhs, conds }
}

use Cond::*;
use Group::*;

pub const SPECS: &[Spec] = &[
    // ---- exact --------------------------------------------------------------------------
    r("comm-add", Exact, "(+ ?a ?b)", "(+ ?b ?a)", &[]),
    r("comm-mul", Exact, "(* ?a ?b)", "(* ?b ?a)", &[]),
    r("comm-min", Exact, "(min ?a ?b)", "(min ?b ?a)", &[]),
    r("comm-max", Exact, "(max ?a ?b)", "(max ?b ?a)", &[]),
    r("mul-one", Exact, "(* ?a #1)", "?a", &[]),
    r("add-zero", Exact, "(+ ?a #0)", "?a", &[]),
    r("sub-zero", Exact, "(- ?a #0)", "?a", &[]),
    r("div-one", Exact, "(/ ?a #1)", "?a", &[]),
    r("neg-neg", Exact, "(neg (neg ?a))", "?a", &[]),
    r("sub-neg", Exact, "(- ?a (neg ?b))", "(+ ?a ?b)", &[]),
    r("add-neg", Exact, "(+ ?a (neg ?b))", "(- ?a ?b)", &[]),
    r("neg-sub", Exact, "(neg (- ?a ?b))", "(- ?b ?a)", &[]),
    r("mul-neg", Exact, "(* (neg ?a) ?b)", "(neg (* ?a ?b))", &[]),
    r("mul-minus-one", Exact, "(* ?a #-1)", "(neg ?a)", &[]),
    r("abs-neg", Exact, "(abs (neg ?a))", "(abs ?a)", &[]),
    r("abs-abs", Exact, "(abs (abs ?a))", "(abs ?a)", &[]),
    r("abs-mul", Exact, "(* (abs ?a) (abs ?b))", "(abs (* ?a ?b))", &[]),
    r("min-self", Exact, "(min ?a ?a)", "?a", &[]),
    r("max-self", Exact, "(max ?a ?a)", "?a", &[]),
    r("min-assoc", Exact, "(min (min ?a ?b) ?c)", "(min ?a (min ?b ?c))", &[]),
    r("max-assoc", Exact, "(max (max ?a ?b) ?c)", "(max ?a (max ?b ?c))", &[]),
    r("min-neg", Exact, "(min (neg ?a) (neg ?b))", "(neg (max ?a ?b))", &[]),
    r("sel-same", Exact, "(sel ?c ?a ?a)", "?a", &[]),
    r("sel-true", Exact, "(sel true ?a ?b)", "?a", &[]),
    r("sel-false", Exact, "(sel false ?a ?b)", "?b", &[]),
    r("sat-sat", Exact, "(saturate (saturate ?a))", "(saturate ?a)", &[]),
    r("clamp-unit", Exact, "(clamp ?a #0 #1)", "(saturate ?a)", &[]),
    r("double", Exact, "(+ ?a ?a)", "(* ?a #2)", &[]),
    r("half", Exact, "(/ ?a #2)", "(* ?a #0.5)", &[]),
    r("quarter", Exact, "(/ ?a #4)", "(* ?a #0.25)", &[]),
    r("floor-floor", Exact, "(floor (floor ?a))", "(floor ?a)", &[]),
    r("length-neg", Exact, "(length (neg ?a))", "(length ?a)", &[]),
    r("dot-neg", Exact, "(dot (neg ?a) ?b)", "(neg (dot ?a ?b))", &[]),
    // The splat family (`(op (splatN a) (splatN b))` to `(splatN (op a b))`, and a component of
    // a splat) is generated: see `splat_rules`.
    // ---- exact given an interval ----------------------------------------------------------
    r("abs-nonneg", Interval, "(abs ?a)", "?a", &[NonNeg("a")]),
    r("abs-nonpos", Interval, "(abs ?a)", "(neg ?a)", &[NonPos("a")]),
    r("max-above", Interval, "(max ?a ?b)", "?a", &[Ge("a", "b")]),
    r("min-below", Interval, "(min ?a ?b)", "?a", &[Le("a", "b")]),
    r("clamp-within", Interval, "(clamp ?a ?l ?h)", "?a", &[In("a", "l", "h")]),
    r("sat-within", Interval, "(saturate ?a)", "?a", &[Unit("a")]),
    r("sqrt-square", Interval, "(sqrt (* ?a ?a))", "(abs ?a)", &[Bounded("a")]),
    r("mul-zero", Interval, "(* ?a #0)", "#0", &[Clean("a")]),
    r("sub-self", Interval, "(- ?a ?a)", "#0", &[Clean("a")]),
    r("lt-true", Interval, "(< ?a ?b)", "true", &[Lt("a", "b")]),
    r("lt-false", Interval, "(< ?a ?b)", "false", &[Ge("a", "b")]),
    r("sign-pos", Interval, "(sign ?a)", "#1", &[Pos("a")]),
    r("step-one", Interval, "(step ?e ?x)", "#1", &[Ge("x", "e")]),
    // ---- within a few ulps ----------------------------------------------------------------
    r("div-rate", Approx, "(/ ?a ?b)", "(* ?a (/ #1 ?b))", &[Slower("b", "a")]),
    r("recip-sqrt", Approx, "(/ #1 (sqrt ?a))", "(isqrt ?a)", &[]),
    r("div-sqrt", Approx, "(/ ?a (sqrt ?b))", "(* ?a (isqrt ?b))", &[]),
    r("sqrt-isqrt", Approx, "(* ?a (isqrt ?a))", "(sqrt ?a)", &[Pos("a")]),
    r("length-normalize", Approx, "(length (normalize ?a))", "1", &[]),
    r("normalize-normalize", Approx, "(normalize (normalize ?a))", "(normalize ?a)", &[]),
    r("length-squared", Approx, "(* (length ?a) (length ?a))", "(dot ?a ?a)", &[]),
    r("sqrt-dot", Approx, "(sqrt (dot ?a ?a))", "(length ?a)", &[]),
    r("distance", Approx, "(distance ?a ?b)", "(length (- ?a ?b))", &[]),
    r("pow-two", Approx, "(pow ?a #2)", "(* ?a ?a)", &[]),
    r("pow-half", Approx, "(pow ?a #0.5)", "(sqrt ?a)", &[]),
    r("pow-one", Approx, "(pow ?a #1)", "?a", &[]),
    r("exp-add", Approx, "(* (exp ?a) (exp ?b))", "(exp (+ ?a ?b))", &[]),
    r("exp2-add", Approx, "(* (exp2 ?a) (exp2 ?b))", "(exp2 (+ ?a ?b))", &[]),
    r("log2-div", Approx, "(log2 (/ ?a ?b))", "(- (log2 ?a) (log2 ?b))", &[NonNeg("a"), Pos("b")]),
    r("log2-mul", Approx, "(log2 (* ?a ?b))", "(+ (log2 ?a) (log2 ?b))", &[NonNeg("a"), Pos("b")]),
    r(
        "sqrt-mul",
        Approx,
        "(* (sqrt ?a) (sqrt ?b))",
        "(sqrt (* ?a ?b))",
        &[NonNeg("a"), NonNeg("b"), Bounded("a"), Bounded("b")],
    ),
    r("mix-same", Approx, "(mix ?a ?a ?t)", "?a", &[]),
    r("mix-zero", Approx, "(mix ?a ?b #0)", "?a", &[]),
    r("mix-one", Approx, "(mix ?a ?b #1)", "?b", &[]),
    r("normalize-scale", Approx, "(normalize (* ?a (splat3 ?s)))", "(normalize ?a)", &[Pos("s")]),
    r("length-scale", Approx, "(length (* ?a (splat3 ?s)))", "(* (length ?a) (abs ?s))", &[]),
    r("dot-scale", Approx, "(dot (* ?a (splat3 ?s)) ?b)", "(* (dot ?a ?b) ?s)", &[]),
    // ---- reassociating (flagged) -----------------------------------------------------------
    r("add-assoc", Reassoc, "(+ (+ ?a ?b) ?c)", "(+ ?a (+ ?b ?c))", &[]),
    r("add-assoc-r", Reassoc, "(+ ?a (+ ?b ?c))", "(+ (+ ?a ?b) ?c)", &[]),
    r("mul-assoc", Reassoc, "(* (* ?a ?b) ?c)", "(* ?a (* ?b ?c))", &[]),
    r("mul-assoc-r", Reassoc, "(* ?a (* ?b ?c))", "(* (* ?a ?b) ?c)", &[]),
    r("factor", Reassoc, "(+ (* ?a ?b) (* ?a ?c))", "(* ?a (+ ?b ?c))", &[]),
    r("factor-sub", Reassoc, "(- (* ?a ?b) (* ?a ?c))", "(* ?a (- ?b ?c))", &[]),
    r("add-sub", Reassoc, "(- (+ ?a ?b) ?b)", "?a", &[]),
    r("sub-assoc", Reassoc, "(- (+ ?a ?b) ?c)", "(+ ?a (- ?b ?c))", &[]),
    r("mul-div", Reassoc, "(/ (* ?a ?b) ?c)", "(* ?a (/ ?b ?c))", &[]),
];

/// A spec's patterns, one pair for each width its `#` constants take (one pair if it has none):
/// the name (with `/w` for a width), the left and right sides.
pub fn expand(spec: &Spec) -> Vec<(String, String, String, u8)> {
    if !spec.lhs.contains('#') && !spec.rhs.contains('#') {
        return vec![(spec.name.to_string(), spec.lhs.to_string(), spec.rhs.to_string(), 0)];
    }
    (1..=4u8)
        .map(|w| {
            let at = |s: &str| widths(s, w);
            (format!("{}/{w}", spec.name), at(spec.lhs), at(spec.rhs), w)
        })
        .collect()
}

fn widths(s: &str, w: u8) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '#' {
            out.push(c);
            continue;
        }
        let mut num = String::new();
        while let Some(&d) = chars.peek() {
            if d == ' ' || d == ')' {
                break;
            }
            num.push(d);
            chars.next();
        }
        if w == 1 { out.push_str(&num) } else { out.push_str(&format!("(splat{w} {num})")) }
    }
    out
}

/// The splat family: `(op (splatN a) (splatN b))` to `(splatN (op a b))` for the arithmetic and
/// `min`/`max`, and a component of a splat. Exact.
fn splat_rules() -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for n in 2..=4 {
        for op in ["+", "-", "*", "/", "min", "max"] {
            out.push((
                format!("splat-op{op}/{n}"),
                format!("({op} (splat{n} ?a) (splat{n} ?b))"),
                format!("(splat{n} ({op} ?a ?b))"),
            ));
        }
        for k in 0..n {
            out.push((
                format!("ext-splat{k}/{n}"),
                format!("(ext{k} (splat{n} ?a))"),
                "?a".to_string(),
            ));
        }
    }
    out
}

/// Which groups to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Groups {
    pub approx: bool,
    pub reassoc: bool,
}

impl Groups {
    pub fn has(self, g: Group) -> bool {
        match g {
            Exact | Interval => true,
            Approx => self.approx,
            Reassoc => self.reassoc,
        }
    }
}

/// The rewrites for these groups.
pub fn rewrites(groups: Groups) -> Vec<Rewrite<Node, crate::analysis::Facts>> {
    let mut out = Vec::new();
    for spec in SPECS.iter().filter(|s| groups.has(s.group)) {
        for (name, lhs, rhs, _) in expand(spec) {
            out.push(make(&name, &lhs, &rhs, spec.conds, spec.group));
        }
    }
    for (name, lhs, rhs) in splat_rules() {
        out.push(make(&name, &lhs, &rhs, &[], Exact));
    }
    out
}

fn make(
    name: &str,
    lhs: &str,
    rhs: &str,
    conds: &'static [Cond],
    group: Group,
) -> Rewrite<Node, crate::analysis::Facts> {
    let lhs: Pattern<Node> = lhs.parse().unwrap_or_else(|e| panic!("rule {name}: {lhs}: {e:?}"));
    let rhs: Pattern<Node> = rhs.parse().unwrap_or_else(|e| panic!("rule {name}: {rhs}: {e:?}"));
    // A comparison's class is a bool; every other rule rewrites floats.
    let float_class = !matches!(name.split('/').next(), Some("lt-true" | "lt-false"));
    let guarded = Guarded { rhs, conds, float_class, group };
    Rewrite::new(name.to_string(), lhs, guarded).unwrap_or_else(|e| panic!("rule {name}: {e}"))
}

/// A rule's right side, applied where its facts hold and its type is the matched class's.
struct Guarded {
    rhs: Pattern<Node>,
    conds: &'static [Cond],
    float_class: bool,
    #[allow(dead_code)]
    group: Group,
}

fn var(name: &str) -> Var {
    format!("?{name}").parse().expect("a variable")
}

fn data<'g>(eg: &'g Graph, subst: &Subst, name: &str) -> &'g Data {
    &eg[subst[var(name)]].data
}

fn clean_iv(d: &Data) -> Option<(f64, f64)> {
    if d.nan { None } else { d.iv }
}

/// Whether a fact holds of the matched classes.
pub fn holds(eg: &Graph, subst: &Subst, c: Cond) -> bool {
    let d = |n| data(eg, subst, n);
    match c {
        NonNeg(a) => d(a).iv.is_some_and(|(lo, _)| lo >= 0.0),
        NonPos(a) => d(a).iv.is_some_and(|(_, hi)| hi <= 0.0),
        Pos(a) => d(a).iv.is_some_and(|(lo, _)| lo > 0.0),
        Clean(a) => clean_iv(d(a)).is_some_and(|(lo, hi)| lo.is_finite() && hi.is_finite()),
        Ge(a, b) => clean_iv(d(a)).zip(clean_iv(d(b))).is_some_and(|(x, y)| x.0 >= y.1),
        Lt(a, b) => clean_iv(d(a)).zip(clean_iv(d(b))).is_some_and(|(x, y)| x.1 < y.0),
        Le(a, b) => clean_iv(d(a)).zip(clean_iv(d(b))).is_some_and(|(x, y)| x.1 <= y.0),
        In(x, l, h) => match (clean_iv(d(x)), clean_iv(d(l)), clean_iv(d(h))) {
            (Some(x), Some(l), Some(h)) => x.0 >= l.1 && x.1 <= h.0,
            _ => false,
        },
        Unit(a) => clean_iv(d(a)).is_some_and(|(lo, hi)| lo >= 0.0 && hi <= 1.0),
        Bounded(a) => d(a).iv.is_some_and(|(lo, hi)| lo >= -BOUND && hi <= BOUND),
        Slower(b, a) => d(b).rate < d(a).rate && d(b).rate != Rate::Const,
    }
}

/// The type a pattern's instance would have (`None`: ill-typed).
fn pattern_type(eg: &mut Graph, ast: &PatternAst<Node>, subst: &Subst) -> Option<TypeId> {
    let mut tys: Vec<TypeId> = Vec::with_capacity(ast.len());
    for n in ast.as_ref() {
        let t = match n {
            ENodeOrVar::Var(v) => eg[subst[*v]].data.ty,
            ENodeOrVar::ENode(e) => {
                let kids: Vec<TypeId> = e.children().iter().map(|&c| tys[usize::from(c)]).collect();
                let crate::analysis::Facts { types, given } = &mut eg.analysis;
                type_of(types, given, e, &kids)?
            }
        };
        tys.push(t);
    }
    tys.last().copied()
}

use egg::Language;

impl Applier<Node, crate::analysis::Facts> for Guarded {
    fn apply_one(
        &self,
        eg: &mut Graph,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Node>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let want = eg[eclass].data.ty;
        if self.float_class
            && !shape(&eg.analysis.types, want).is_some_and(|(s, _)| s == Scalar::F32)
        {
            return vec![];
        }
        if !self.conds.iter().all(|&c| holds(eg, subst, c)) {
            return vec![];
        }
        if pattern_type(eg, &self.rhs.ast, subst) != Some(want) {
            return vec![];
        }
        self.rhs.apply_one(eg, eclass, subst, searcher_ast, rule_name)
    }

    fn vars(&self) -> Vec<Var> {
        self.rhs.vars()
    }
}
