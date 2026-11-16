//! The e-graph's language: the IR's pure expressions, one node each, with their operands as
//! e-class ids. What a rewrite can't see into (a load, a texture read, a conversion, the
//! invocation's inputs) is an [`Node::Opaque`] node: kept as written, its operands still classes.

use egg::{FromOp, Id, Language};
use std::fmt;
use wrela_ir::{BinOp, Builtin, Const, Scalar, UnOp};

/// A scalar literal: its scalar type and its bits (a float's `to_bits`, an integer widened, a
/// bool 0 or 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Lit {
    pub s: u8,
    pub bits: u64,
}

impl Lit {
    pub fn f32(x: f32) -> Lit {
        Lit { s: scalar_code(Scalar::F32), bits: u64::from(x.to_bits()) }
    }
    pub fn bool(b: bool) -> Lit {
        Lit { s: scalar_code(Scalar::Bool), bits: u64::from(b) }
    }
    pub fn scalar(self) -> Scalar {
        SCALARS[self.s as usize]
    }
    pub fn as_f32(self) -> Option<f32> {
        (self.scalar() == Scalar::F32).then(|| f32::from_bits(self.bits as u32))
    }
    pub fn from_const(c: &Const) -> Option<Lit> {
        let (s, bits) = match *c {
            Const::Bool(b) => (Scalar::Bool, u64::from(b)),
            Const::I32(x) => (Scalar::I32, u64::from(x as u32)),
            Const::U32(x) => (Scalar::U32, u64::from(x)),
            Const::F32(x) => (Scalar::F32, u64::from(x.to_bits())),
            _ => return None,
        };
        Some(Lit { s: scalar_code(s), bits })
    }
    pub fn to_const(self) -> Const {
        match self.scalar() {
            Scalar::Bool => Const::Bool(self.bits != 0),
            Scalar::I32 => Const::I32(self.bits as u32 as i32),
            Scalar::U32 => Const::U32(self.bits as u32),
            Scalar::F32 => Const::F32(f32::from_bits(self.bits as u32)),
            s => unreachable!("no literal of {s:?}"),
        }
    }
}

const SCALARS: [Scalar; 4] = [Scalar::Bool, Scalar::I32, Scalar::U32, Scalar::F32];

fn scalar_code(s: Scalar) -> u8 {
    SCALARS.iter().position(|&x| x == s).expect("a literal's scalar") as u8
}

/// The binary operators, in an order (egg wants its nodes ordered).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Bop(pub u8);

const BINOPS: [BinOp; 21] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::Div,
    BinOp::Rem,
    BinOp::WrappingAdd,
    BinOp::WrappingSub,
    BinOp::WrappingMul,
    BinOp::Shl,
    BinOp::Shr,
    BinOp::BitAnd,
    BinOp::BitOr,
    BinOp::BitXor,
    BinOp::And,
    BinOp::Or,
    BinOp::Eq,
    BinOp::Ne,
    BinOp::Lt,
    BinOp::Le,
    BinOp::Gt,
    BinOp::Ge,
];

impl Bop {
    pub fn of(op: BinOp) -> Bop {
        Bop(BINOPS.iter().position(|&x| x == op).expect("every BinOp is listed") as u8)
    }
    pub fn op(self) -> BinOp {
        BINOPS[self.0 as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Uop {
    Neg,
    Not,
}

impl Uop {
    pub fn of(op: UnOp) -> Uop {
        match op {
            UnOp::Neg => Uop::Neg,
            UnOp::Not => Uop::Not,
        }
    }
    pub fn op(self) -> UnOp {
        match self {
            Uop::Neg => UnOp::Neg,
            Uop::Not => UnOp::Not,
        }
    }
}

/// One e-node.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Node {
    /// A value from outside the stretch (by its IR value number).
    Leaf(u32),
    Lit(Lit),
    Bin(Bop, [Id; 2]),
    Un(Uop, [Id; 1]),
    Call(Builtin, Box<[Id]>),
    Splat(u8, [Id; 1]),
    Swizzle(Box<[u8]>, [Id; 1]),
    Extract(u32, [Id; 1]),
    /// A construct of the type with this IR type number.
    Construct(u32, Box<[Id]>),
    /// cond, if_true, if_false.
    Select([Id; 3]),
    /// A let of the stretch kept as written (by its IR value number), its operands in the order
    /// `Expr::for_each_value` gives them.
    Opaque(u32, Box<[Id]>),
}

impl Node {
    /// The node with its children zeroed: what it is, not what it's of.
    fn key(&self) -> Node {
        let mut k = self.clone();
        for c in k.children_mut() {
            *c = Id::from(0);
        }
        k
    }
}

impl Language for Node {
    type Discriminant = Node;

    fn discriminant(&self) -> Node {
        self.key()
    }

    fn matches(&self, other: &Self) -> bool {
        self.key() == other.key()
    }

    fn children(&self) -> &[Id] {
        match self {
            Node::Leaf(_) | Node::Lit(_) => &[],
            Node::Bin(_, c) => c,
            Node::Un(_, c) | Node::Splat(_, c) | Node::Swizzle(_, c) | Node::Extract(_, c) => c,
            Node::Call(_, c) | Node::Construct(_, c) | Node::Opaque(_, c) => c,
            Node::Select(c) => c,
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Node::Leaf(_) | Node::Lit(_) => &mut [],
            Node::Bin(_, c) => c,
            Node::Un(_, c) | Node::Splat(_, c) | Node::Swizzle(_, c) | Node::Extract(_, c) => c,
            Node::Call(_, c) | Node::Construct(_, c) | Node::Opaque(_, c) => c,
            Node::Select(c) => c,
        }
    }
}

/// The builtins a pattern can name, and their names there.
pub const BUILTIN_NAMES: &[(Builtin, &str)] = &[
    (Builtin::Sqrt, "sqrt"),
    (Builtin::InverseSqrt, "isqrt"),
    (Builtin::Sin, "sin"),
    (Builtin::Cos, "cos"),
    (Builtin::Exp, "exp"),
    (Builtin::Exp2, "exp2"),
    (Builtin::Log, "log"),
    (Builtin::Log2, "log2"),
    (Builtin::Pow, "pow"),
    (Builtin::Floor, "floor"),
    (Builtin::Ceil, "ceil"),
    (Builtin::Fract, "fract"),
    (Builtin::Abs, "abs"),
    (Builtin::Sign, "sign"),
    (Builtin::Min, "min"),
    (Builtin::Max, "max"),
    (Builtin::Clamp, "clamp"),
    (Builtin::Saturate, "saturate"),
    (Builtin::Mix, "mix"),
    (Builtin::Step, "step"),
    (Builtin::Smoothstep, "smoothstep"),
    (Builtin::Length, "length"),
    (Builtin::Distance, "distance"),
    (Builtin::Dot, "dot"),
    (Builtin::Normalize, "normalize"),
];

const BIN_NAMES: &[(BinOp, &str)] = &[
    (BinOp::Add, "+"),
    (BinOp::Sub, "-"),
    (BinOp::Mul, "*"),
    (BinOp::Div, "/"),
    (BinOp::Lt, "<"),
    (BinOp::Le, "<="),
    (BinOp::Gt, ">"),
    (BinOp::Ge, ">="),
    (BinOp::Eq, "=="),
    (BinOp::Ne, "!="),
];

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Leaf(v) => write!(f, "v{v}"),
            Node::Lit(l) => match l.as_f32() {
                Some(x) => write!(f, "{x:?}"),
                None => write!(f, "{:?}", l.to_const()),
            },
            Node::Bin(op, _) => match BIN_NAMES.iter().find(|(o, _)| *o == op.op()) {
                Some((_, n)) => f.write_str(n),
                None => write!(f, "{:?}", op.op()),
            },
            Node::Un(Uop::Neg, _) => f.write_str("neg"),
            Node::Un(Uop::Not, _) => f.write_str("not"),
            Node::Call(b, _) => match BUILTIN_NAMES.iter().find(|(x, _)| x == b) {
                Some((_, n)) => f.write_str(n),
                None => write!(f, "{b:?}"),
            },
            Node::Splat(n, _) => write!(f, "splat{n}"),
            Node::Swizzle(s, _) => write!(f, "swizzle{s:?}"),
            Node::Extract(k, _) => write!(f, "ext{k}"),
            Node::Construct(t, _) => write!(f, "construct{t}"),
            Node::Select(_) => f.write_str("sel"),
            Node::Opaque(v, _) => write!(f, "opaque{v}"),
        }
    }
}

#[derive(Debug)]
pub struct ParseError(pub String);

impl FromOp for Node {
    type Error = ParseError;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Node, ParseError> {
        let err = || ParseError(format!("`{op}` with {} operands", children.len()));
        let one = |c: &[Id]| -> Result<[Id; 1], ParseError> { c.try_into().map_err(|_| err()) };
        if let Some((b, _)) = BIN_NAMES.iter().find(|(_, n)| *n == op) {
            let c: [Id; 2] = children.as_slice().try_into().map_err(|_| err())?;
            return Ok(Node::Bin(Bop::of(*b), c));
        }
        if let Some((b, _)) = BUILTIN_NAMES.iter().find(|(_, n)| *n == op) {
            if children.len() != b.arity() {
                return Err(err());
            }
            return Ok(Node::Call(*b, children.into_boxed_slice()));
        }
        match op {
            "neg" => return Ok(Node::Un(Uop::Neg, one(&children)?)),
            "sel" => {
                let c: [Id; 3] = children.as_slice().try_into().map_err(|_| err())?;
                return Ok(Node::Select(c));
            }
            "true" | "false" if children.is_empty() => {
                return Ok(Node::Lit(Lit::bool(op == "true")));
            }
            _ => {}
        }
        // A leaf, for tests: `v3`.
        if children.is_empty()
            && let Some(v) = op.strip_prefix('v').and_then(|n| n.parse::<u32>().ok())
        {
            return Ok(Node::Leaf(v));
        }
        if let Some(n) = op.strip_prefix("splat").and_then(|n| n.parse::<u8>().ok()) {
            return Ok(Node::Splat(n, one(&children)?));
        }
        if let Some(k) = op.strip_prefix("ext").and_then(|n| n.parse::<u32>().ok()) {
            return Ok(Node::Extract(k, one(&children)?));
        }
        if children.is_empty()
            && let Ok(x) = op.parse::<f32>()
        {
            return Ok(Node::Lit(Lit::f32(x)));
        }
        Err(err())
    }
}
