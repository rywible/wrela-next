//! What both writers work out of a GPU module before writing it: the resources each function
//! reaches, the functions that stay calls and in what order, which expressions only compute, and
//! what the compiler proves of the run-time checks.

use crate::Checks;
use std::collections::{BTreeSet, HashMap, HashSet};
use wrela_ir as ir;

/// The resources each function uses, directly or through its calls.
pub(crate) fn resource_uses(m: &ir::Module) -> Vec<BTreeSet<usize>> {
    let direct: Vec<BTreeSet<usize>> = m
        .functions
        .iter()
        .map(|f| {
            let mut used = BTreeSet::new();
            ir::visit::walk(&f.body, &mut |s| {
                s.for_each_place(&mut |p| {
                    if let ir::PlaceRoot::Resource(r) = p.root {
                        used.insert(r.index());
                    }
                });
                match s.expr() {
                    Some(ir::Expr::Texture(_, t, s, _)) => {
                        used.insert(t.index());
                        if let Some(s) = s {
                            used.insert(s.index());
                        }
                    }
                    Some(ir::Expr::TextureStore(t, _)) => {
                        used.insert(t.index());
                    }
                    _ => {}
                }
            });
            used
        })
        .collect();
    // (What a function reaches includes itself.)
    (0..m.functions.len())
        .map(|f| {
            let mut all = BTreeSet::new();
            for g in ir::visit::reachable(m, ir::FuncId(f as u32), &[]) {
                all.extend(direct[g.index()].iter().copied());
            }
            all
        })
        .collect()
}

/// The functions that stay calls (every one the entry points reach but the entry points
/// themselves), callees first.
pub(crate) fn helpers(m: &ir::Module) -> Vec<ir::FuncId> {
    fn visit(
        m: &ir::Module,
        f: ir::FuncId,
        entries: &HashSet<ir::FuncId>,
        done: &mut HashSet<ir::FuncId>,
        order: &mut Vec<ir::FuncId>,
    ) {
        if !done.insert(f) {
            return;
        }
        for g in ir::visit::calls(&m.functions[f.index()].body) {
            visit(m, g, entries, done, order);
        }
        if !entries.contains(&f) {
            order.push(f);
        }
    }
    let entries: HashSet<ir::FuncId> = m.entry_points.iter().map(|e| e.function).collect();
    let mut done = HashSet::new();
    let mut order = Vec::new();
    for e in &m.entry_points {
        visit(m, e.function, &entries, &mut done, &mut order);
    }
    order
}

/// Whether `e` only computes from values, and isn't a derivative (which wants the control flow
/// it's in) or an element at a dynamic index: nothing need be done for it if nothing reads it,
/// and Metal writes it where it's used.
pub(crate) fn pure(e: &ir::Expr) -> bool {
    e.only_computes()
        && !matches!(
            e,
            ir::Expr::ExtractDyn(..)
                | ir::Expr::Builtin(ir::Builtin::Dpdx | ir::Builtin::Dpdy | ir::Builtin::Fwidth, _)
        )
}

/// The integer constant `v` is, among a function's `consts` (`ir::counted::constants`).
pub(crate) fn const_int(consts: &HashMap<ir::ValueId, ir::Const>, v: ir::ValueId) -> Option<i64> {
    match consts.get(&v)? {
        ir::Const::I32(x) => Some(i64::from(*x)),
        ir::Const::U32(x) => Some(i64::from(*x)),
        _ => None,
    }
}

/// How a payload-less enum (`TypeDef::Enum` whose variants all carry `None`) is lowered to a
/// scalar in code: a two-variant `[0, 1]` enum (such as `Option<EmptyProof>`) lowers to `bool`,
/// and any other payload-less enum lowers to its `u32` tag directly (no struct wrapper).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnitEnum {
    Bool,
    Tag,
}

/// Whether `t` is a payload-less enum and which scalar form it takes.
pub(crate) fn unit_enum(types: &ir::Types, t: ir::TypeId) -> Option<UnitEnum> {
    let ir::TypeDef::Enum { variants, tags, .. } = types.get(t) else { return None };
    if !variants.iter().all(|(_, p)| p.is_none()) {
        return None;
    }
    let bool_tags = variants.len() == 2 && (tags.is_empty() || tags.as_slice() == [0, 1]);
    Some(if bool_tags { UnitEnum::Bool } else { UnitEnum::Tag })
}

/// What the compiler proves of `m`'s run-time checks (`wrela_ir::work`), where `checks` writes
/// any in: the ones it makes unneeded are left out.
pub(crate) fn proofs(m: &ir::Module, checks: Checks) -> ir::work::Proofs {
    if checks.unchecked() { ir::work::Proofs::default() } else { ir::work::proofs(m) }
}
