//! What an extraction costs: language.md §18.11's work weights, scaled by how often the work
//! happens. Work the same for the whole command costs almost nothing (step 4 would move it out
//! of the shader), and a constant's nothing at all.

use crate::analysis::{Graph, Rate, shape};
use crate::lang::Node;
use egg::{CostFunction, Id, Language};
use wrela_ir::{BinOp, Builtin};

/// A unit of work the same for every invocation, against one that varies.
pub const UNIFORM: f64 = 0.01;
/// A unit of constant work (folded by the analysis where it's exact).
pub const CONST: f64 = 0.001;
/// Each node, so that of two forms of equal work the smaller wins.
const NODE: f64 = 1e-6;

fn comps(eg: &Graph, c: Id) -> u64 {
    shape(&eg.analysis.types, eg[c].data.ty).map_or(1, |(_, n)| u64::from(n))
}

/// A node's own work (§18.11), not scaled by its rate.
pub fn own_work(eg: &Graph, n: &Node) -> f64 {
    let kids = n.children();
    match n {
        Node::Leaf(_) | Node::Lit(_) => 0.0,
        Node::Opaque(v, _) => eg.analysis.given[v].work,
        Node::Bin(op, [a, b]) => {
            let per = if matches!(op.op(), BinOp::Div | BinOp::Rem) { 4 } else { 1 };
            (per * comps(eg, *a).max(comps(eg, *b))) as f64
        }
        Node::Un(_, [a]) => comps(eg, *a) as f64,
        Node::Call(b, _) => wrela_ir::work::builtin_cost(*b, comps(eg, kids[0])) as f64,
        Node::Splat(..) | Node::Swizzle(..) | Node::Extract(..) | Node::Construct(..) => 0.0,
        Node::Select([_, t, _]) => comps(eg, *t) as f64,
    }
}

/// How often a node's own work happens: its operands' rate (a leaf's or kept let's own).
pub fn node_rate(eg: &Graph, n: &Node) -> Rate {
    match n {
        Node::Leaf(v) | Node::Opaque(v, _) => eg.analysis.given[v].rate,
        Node::Lit(_) => Rate::Const,
        // A derivative of a uniform value is zero, but it's taken per invocation all the same.
        Node::Call(Builtin::Dpdx | Builtin::Dpdy | Builtin::Fwidth, _) => Rate::Varying,
        _ => n.children().iter().map(|&c| eg[c].data.rate).max().unwrap_or(Rate::Const),
    }
}

pub fn scale(r: Rate) -> f64 {
    match r {
        Rate::Const => CONST,
        Rate::Uniform => UNIFORM,
        Rate::Varying => 1.0,
    }
}

/// A node's own cost: its work at its rate.
pub fn own_cost(eg: &Graph, n: &Node) -> f64 {
    own_work(eg, n) * scale(node_rate(eg, n)) + NODE
}

/// The extraction's cost: a tree's own costs, summed.
pub struct Cost<'a>(pub &'a Graph);

impl CostFunction<Node> for Cost<'_> {
    type Cost = f64;

    fn cost<C>(&mut self, n: &Node, mut costs: C) -> f64
    where
        C: FnMut(Id) -> f64,
    {
        own_cost(self.0, n) + n.children().iter().map(|&c| costs(c)).sum::<f64>()
    }
}
