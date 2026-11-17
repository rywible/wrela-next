//! Uniform work moved out of the shader (round 2 of the spike): a class the same for the whole
//! command whose value comes only from the uniform block and constants is written as a tape
//! (`wrela_abi::hoist`) the host runs once a command, and the shader reads its value from a
//! field appended to its uniform block.

use crate::analysis::shape;
use std::collections::HashMap;
use wrela_abi::hoist::{Fun, Kind, Op, Output, Tape};
use wrela_ir::{
    BinOp, Builtin, Const, Expr, Module, Place, PlaceRoot, Proj, ResourceId, ResourceKind, Scalar,
    TypeDef, TypeId, Types, UnOp, ValueId,
};

/// A value on the tape, or a place in the uniform bytes not yet read.
#[derive(Clone, Debug)]
pub enum Tv {
    Val { op: u32, ty: TypeId, key: String },
    Mem { offset: u32, ty: TypeId, key: String },
}

impl Tv {
    pub fn ty(&self) -> TypeId {
        match self {
            Tv::Val { ty, .. } | Tv::Mem { ty, .. } => *ty,
        }
    }
    pub fn key(&self) -> &str {
        match self {
            Tv::Val { key, .. } | Tv::Mem { key, .. } => key,
        }
    }
}

/// What a value is made of, its operands already on the tape.
pub enum Term {
    Lit(Const),
    Bin(BinOp, Tv, Tv),
    Un(UnOp, Tv),
    Call(Builtin, Vec<Tv>),
    Splat(u8, Tv),
    Swizzle(Vec<u8>, Tv),
    Extract(u32, Tv),
    Construct(TypeId, Vec<Tv>),
    Select(Tv, Tv, Tv),
    Convert(Tv, Scalar),
    /// A read of the uniform block at a place whose indices are these constants.
    Load(Place, Vec<u32>),
}

pub struct Hoist {
    pub res: ResourceId,
    pub orig: TypeId,
    base_fields: u32,
    pub fields: Vec<(String, TypeId)>,
    slots: HashMap<String, u32>,
    ops: Vec<Op>,
    memo: HashMap<String, u32>,
    outputs: Vec<(u32, u32)>,
    /// The work moved out of the shader, as §18.11 counts it at each place it was read.
    pub moved_work: f64,
    pub reads: usize,
}

fn kind(types: &Types, t: TypeId) -> Option<(Kind, u8)> {
    let (s, n) = shape(types, t)?;
    let k = match s {
        Scalar::F32 => Kind::F32,
        Scalar::U32 => Kind::U32,
        Scalar::I32 => Kind::I32,
        Scalar::Bool => Kind::Bool,
        _ => return None,
    };
    Some((k, n))
}

fn fun_of_bin(op: BinOp) -> Fun {
    use BinOp as B;
    match op {
        B::Add | B::WrappingAdd => Fun::Add,
        B::Sub | B::WrappingSub => Fun::Sub,
        B::Mul | B::WrappingMul => Fun::Mul,
        B::Div => Fun::Div,
        B::Rem => Fun::Rem,
        B::Shl => Fun::Shl,
        B::Shr => Fun::Shr,
        B::BitAnd => Fun::BitAnd,
        B::BitOr => Fun::BitOr,
        B::BitXor => Fun::BitXor,
        B::And => Fun::And,
        B::Or => Fun::Or,
        B::Eq => Fun::Eq,
        B::Ne => Fun::Ne,
        B::Lt => Fun::Lt,
        B::Le => Fun::Le,
        B::Gt => Fun::Gt,
        B::Ge => Fun::Ge,
    }
}

fn fun_of_builtin(b: Builtin) -> Option<Fun> {
    use Builtin as B;
    Some(match b {
        B::Sqrt => Fun::Sqrt,
        B::InverseSqrt => Fun::InverseSqrt,
        B::Sin => Fun::Sin,
        B::Cos => Fun::Cos,
        B::Tan => Fun::Tan,
        B::Asin => Fun::Asin,
        B::Acos => Fun::Acos,
        B::Atan => Fun::Atan,
        B::Atan2 => Fun::Atan2,
        B::Exp => Fun::Exp,
        B::Exp2 => Fun::Exp2,
        B::Log => Fun::Log,
        B::Log2 => Fun::Log2,
        B::Sinh => Fun::Sinh,
        B::Cosh => Fun::Cosh,
        B::Tanh => Fun::Tanh,
        B::Pow => Fun::Pow,
        B::Floor => Fun::Floor,
        B::Ceil => Fun::Ceil,
        B::Round => Fun::Round,
        B::Trunc => Fun::Trunc,
        B::Fract => Fun::Fract,
        B::Abs => Fun::Abs,
        B::Sign => Fun::Sign,
        B::Min => Fun::Min,
        B::Max => Fun::Max,
        B::Clamp => Fun::Clamp,
        B::Saturate => Fun::Saturate,
        B::Mix => Fun::Mix,
        B::Step => Fun::Step,
        B::Smoothstep => Fun::Smoothstep,
        B::Length => Fun::Length,
        B::Distance => Fun::Distance,
        B::Dot => Fun::Dot,
        B::Cross => Fun::Cross,
        B::Normalize => Fun::Normalize,
        _ => return None,
    })
}

impl Hoist {
    /// For a module with a uniform block.
    pub fn new(m: &Module) -> Option<Hoist> {
        let (i, r) = m
            .resources
            .iter()
            .enumerate()
            .find(|(_, r)| matches!(r.kind, ResourceKind::Uniform { .. }))?;
        let TypeDef::Struct { fields, .. } = m.types.get(r.ty) else { return None };
        Some(Hoist {
            res: ResourceId(i as u32),
            orig: r.ty,
            base_fields: fields.len() as u32,
            fields: Vec::new(),
            slots: HashMap::new(),
            ops: Vec::new(),
            memo: HashMap::new(),
            outputs: Vec::new(),
            moved_work: 0.0,
            reads: 0,
        })
    }

    fn push(&mut self, key: String, op: Op, ty: TypeId) -> Tv {
        if let Some(&i) = self.memo.get(&key) {
            return Tv::Val { op: i, ty, key };
        }
        self.ops.push(op);
        let i = self.ops.len() as u32 - 1;
        self.memo.insert(key.clone(), i);
        Tv::Val { op: i, ty, key }
    }

    /// A tape value's operation, reading a place if it's one.
    fn val(&mut self, types: &Types, t: &Tv) -> Option<u32> {
        match t {
            Tv::Val { op, .. } => Some(*op),
            Tv::Mem { offset, ty, key } => {
                let (k, n) = kind(types, *ty)?;
                if k == Kind::Bool {
                    return None;
                }
                match self.push(
                    format!("load({key})"),
                    Op::Load { offset: *offset, kind: k, comps: n },
                    *ty,
                ) {
                    Tv::Val { op, .. } => Some(op),
                    Tv::Mem { .. } => None,
                }
            }
        }
    }

    /// The tape value of a term of type `ty`, if the host can compute it.
    pub fn term(&mut self, types: &Types, t: Term, ty: TypeId) -> Option<Tv> {
        let tys = types;
        Some(match t {
            Term::Lit(c) => {
                let (k, bits) = match c {
                    Const::F32(x) => (Kind::F32, x.to_bits()),
                    Const::U32(x) => (Kind::U32, x),
                    Const::I32(x) => (Kind::I32, x as u32),
                    Const::Bool(b) => (Kind::Bool, u32::from(b)),
                    _ => return None,
                };
                self.push(format!("c{k:?}{bits:x}"), Op::Const { kind: k, bits }, ty)
            }
            Term::Load(p, idx) => {
                let mut t = self.orig;
                let mut offset = 0u32;
                let mut idx = idx.into_iter();
                let mut key = String::from("u");
                for proj in &p.path {
                    match (tys.get(t).clone(), proj) {
                        (TypeDef::Struct { fields, .. }, Proj::Field(k)) => {
                            offset += wrela_ir::layout::field_offsets(tys, t)[*k as usize];
                            t = fields[*k as usize].1;
                            key.push_str(&format!(".{k}"));
                        }
                        (TypeDef::Vector(s, _), Proj::Comp(c)) => {
                            offset += 4 * u32::from(*c);
                            t = tys.lookup(&TypeDef::Scalar(s))?;
                            key.push_str(&format!(".c{c}"));
                        }
                        (TypeDef::Array(e, n), Proj::Index(_)) => {
                            let i = idx.next()?;
                            if i >= n {
                                return None;
                            }
                            offset += i * wrela_ir::layout::layout(tys, e).stride();
                            t = e;
                            key.push_str(&format!("[{i}]"));
                        }
                        _ => return None,
                    }
                }
                Tv::Mem { offset, ty: t, key }
            }
            Term::Extract(k, a) => match a {
                Tv::Mem { offset, ty: at, key } => match tys.get(at).clone() {
                    TypeDef::Struct { fields, .. } => Tv::Mem {
                        offset: offset + wrela_ir::layout::field_offsets(tys, at)[k as usize],
                        ty: fields[k as usize].1,
                        key: format!("{key}.{k}"),
                    },
                    TypeDef::Vector(s, _) => Tv::Mem {
                        offset: offset + 4 * k,
                        ty: tys.lookup(&TypeDef::Scalar(s))?,
                        key: format!("{key}.c{k}"),
                    },
                    TypeDef::Array(e, n) if k < n => Tv::Mem {
                        offset: offset + k * wrela_ir::layout::layout(tys, e).stride(),
                        ty: e,
                        key: format!("{key}[{k}]"),
                    },
                    _ => return None,
                },
                a => {
                    kind(tys, a.ty())?;
                    let arg = self.val(tys, &a)?;
                    self.push(format!("ext{k}({})", a.key()), Op::Extract { arg, k: k as u8 }, ty)
                }
            },
            Term::Bin(op, a, b) => {
                kind(tys, ty)?;
                let f = fun_of_bin(op);
                let (x, y) = (self.val(tys, &a)?, self.val(tys, &b)?);
                self.push(
                    format!("{f:?}({},{})", a.key(), b.key()),
                    Op::Apply { f, args: vec![x, y] },
                    ty,
                )
            }
            Term::Un(op, a) => {
                kind(tys, ty)?;
                let f = if op == UnOp::Neg { Fun::Neg } else { Fun::Not };
                let x = self.val(tys, &a)?;
                self.push(format!("{f:?}({})", a.key()), Op::Apply { f, args: vec![x] }, ty)
            }
            Term::Call(b, args) => {
                kind(tys, ty)?;
                let f = fun_of_builtin(b)?;
                let mut ids = Vec::new();
                for a in &args {
                    ids.push(self.val(tys, a)?);
                }
                let keys: Vec<&str> = args.iter().map(|a| a.key()).collect();
                self.push(format!("{f:?}({})", keys.join(",")), Op::Apply { f, args: ids }, ty)
            }
            Term::Splat(n, a) => {
                let x = self.val(tys, &a)?;
                self.push(format!("splat{n}({})", a.key()), Op::Splat { arg: x, n }, ty)
            }
            Term::Swizzle(idx, a) => {
                let x = self.val(tys, &a)?;
                self.push(format!("swz{idx:?}({})", a.key()), Op::Swizzle { arg: x, idx }, ty)
            }
            Term::Construct(t, args) => {
                kind(tys, t)?;
                let mut ids = Vec::new();
                for a in &args {
                    ids.push(self.val(tys, a)?);
                }
                let keys: Vec<&str> = args.iter().map(|a| a.key()).collect();
                self.push(format!("vec({})", keys.join(",")), Op::Construct { args: ids }, ty)
            }
            Term::Select(c, a, b) => {
                kind(tys, ty)?;
                let (cc, x, y) = (self.val(tys, &c)?, self.val(tys, &a)?, self.val(tys, &b)?);
                self.push(
                    format!("sel({},{},{})", c.key(), a.key(), b.key()),
                    Op::Select { cond: cc, if_true: x, if_false: y },
                    ty,
                )
            }
            Term::Convert(a, s) => {
                let (k, _) = kind(tys, ty)?;
                let x = self.val(tys, &a)?;
                self.push(format!("to{s:?}({})", a.key()), Op::Convert { arg: x, to: k }, ty)
            }
        })
    }

    /// The field that holds `t` for the shader (a bool's as a `u32`): its index in the
    /// extended block, and whether it's a bool read as a `u32`.
    pub fn slot(&mut self, types: &Types, t: &Tv) -> Option<(u32, bool)> {
        let (k, n) = kind(types, t.ty())?;
        if let Some(&f) = self.slots.get(t.key()) {
            return Some((f, k == Kind::Bool));
        }
        let op = self.val(types, t)?;
        let field_ty = if k == Kind::Bool {
            if n != 1 {
                return None;
            }
            types.lookup(&TypeDef::Scalar(Scalar::U32))?
        } else {
            t.ty()
        };
        let f = self.base_fields + self.fields.len() as u32;
        self.fields.push((format!("hoisted{}", self.fields.len()), field_ty));
        self.slots.insert(t.key().to_string(), f);
        self.outputs.push((op, f));
        Some((f, k == Kind::Bool))
    }

    pub fn place(&self, field: u32) -> Place {
        Place { root: PlaceRoot::Resource(self.res), path: vec![Proj::Field(field)] }
    }

    /// Extends the module's uniform block with the hoisted fields; the tape that fills them.
    pub fn finish(self, m: &mut Module) -> Option<Tape> {
        if self.fields.is_empty() {
            return None;
        }
        let TypeDef::Struct { name, fields } = m.types.get(self.orig).clone() else { return None };
        let program_size = wrela_ir::layout::layout(&m.types, self.orig).size;
        let mut all = fields;
        all.extend(self.fields.iter().cloned());
        let t = m.types.intern(TypeDef::Struct { name: format!("{name}_hoisted"), fields: all });
        m.resources[self.res.index()].ty = t;
        let offsets = wrela_ir::layout::field_offsets(&m.types, t).to_vec();
        let size = wrela_ir::layout::round_up(16, wrela_ir::layout::layout(&m.types, t).size);
        let outputs = self
            .outputs
            .iter()
            .map(|&(op, f)| Output { op, offset: offsets[f as usize] })
            .collect();
        Some(Tape { program_size, size, ops: self.ops, outputs })
    }

    /// Reads of values the IR computes before a stretch, as terms (its leaves).
    pub fn value(
        &mut self,
        types: &Types,
        defs: &HashMap<ValueId, Expr>,
        f: &wrela_ir::Function,
        v: ValueId,
        depth: u32,
    ) -> Option<Tv> {
        if depth > 64 {
            return None;
        }
        let e = defs.get(&v)?.clone();
        let mut ops = Vec::new();
        let mut ok = true;
        e.for_each_value(&mut |x| ops.push(x));
        let consts: Vec<u32> = match &e {
            Expr::Load(p) => {
                let mut out = Vec::new();
                for proj in &p.path {
                    if let Proj::Index(x) = proj {
                        match defs.get(x) {
                            Some(Expr::Const(Const::U32(i))) => out.push(*i),
                            Some(Expr::Const(Const::I32(i))) if *i >= 0 => out.push(*i as u32),
                            _ => ok = false,
                        }
                    }
                }
                out
            }
            _ => Vec::new(),
        };
        if !ok {
            return None;
        }
        let sub =
            |me: &mut Hoist, types: &Types, x: ValueId| me.value(types, defs, f, x, depth + 1);
        let ty = f.value_ty(v);
        let term = match e {
            Expr::Const(c) => Term::Lit(c),
            Expr::Load(p) if p.root == PlaceRoot::Resource(self.res) => Term::Load(p, consts),
            Expr::Binary(op, a, b) => Term::Bin(op, sub(self, types, a)?, sub(self, types, b)?),
            Expr::Unary(op, a) => Term::Un(op, sub(self, types, a)?),
            Expr::Builtin(b, xs) => {
                let mut a = Vec::new();
                for x in xs {
                    a.push(sub(self, types, x)?);
                }
                Term::Call(b, a)
            }
            Expr::Splat(x, n) => Term::Splat(n, sub(self, types, x)?),
            Expr::Swizzle(x, idx) => Term::Swizzle(idx, sub(self, types, x)?),
            Expr::Extract(x, k) => Term::Extract(k, sub(self, types, x)?),
            Expr::Construct(t, xs) => {
                let mut a = Vec::new();
                for x in xs {
                    a.push(sub(self, types, x)?);
                }
                Term::Construct(t, a)
            }
            Expr::Select { cond, if_true, if_false } => Term::Select(
                sub(self, types, cond)?,
                sub(self, types, if_true)?,
                sub(self, types, if_false)?,
            ),
            Expr::Convert(x, s) => Term::Convert(sub(self, types, x)?, s),
            _ => return None,
        };
        self.term(types, term, ty)
    }
}
