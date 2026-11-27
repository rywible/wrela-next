//! Deterministic, bit-exact content fingerprinting of an [`ir::Module`].
//!
//! Feeds every semantically relevant part of a module — types, resources, functions (including
//! exact float bit patterns, spans and budgets), entry points, constant data, and mode flags —
//! into a byte sink so callers can hash a pre-optimization GPU module directly into SHA-256
//! without intermediate allocations.

use crate::*;

pub fn feed_module(m: &Module, feed: &mut impl FnMut(&[u8])) {
    let mut w = Sink(feed);
    w.tag(1); // format version
    w.bool(m.exact);
    w.bool(m.strict);
    w.bool(m.rounded_sqrt);
    w.bool(m.subgroups);

    let types: Vec<&TypeDef> = m.types.iter().map(|(_, d)| d).collect();
    w.len(types.len());
    for d in types {
        feed_type_def(&mut w, d);
    }

    w.len(m.resources.len());
    for r in &m.resources {
        w.str(&r.name);
        w.u32(r.binding);
        feed_resource_kind(&mut w, r.kind);
        w.u32(r.ty.0);
    }

    w.len(m.entry_points.len());
    for e in &m.entry_points {
        w.str(&e.name);
        feed_stage(&mut w, &e.stage);
        w.u32(e.function.0);
        w.len(e.inputs.len());
        for &inp in &e.inputs {
            feed_builtin_input(&mut w, inp);
        }
    }

    w.len(m.data.len());
    for d in &m.data {
        w.str(&d.name);
        w.u32(d.ty.0);
        feed_const_value(&mut w, &d.value);
    }

    w.len(m.functions.len());
    for f in &m.functions {
        feed_function(&mut w, f);
    }
}

struct Sink<'a, F>(&'a mut F);

impl<F: FnMut(&[u8])> Sink<'_, F> {
    fn bytes(&mut self, b: &[u8]) {
        (self.0)(b);
    }
    fn tag(&mut self, t: u8) {
        self.bytes(&[t]);
    }
    fn bool(&mut self, b: bool) {
        self.tag(u8::from(b));
    }
    fn u32(&mut self, x: u32) {
        self.bytes(&x.to_le_bytes());
    }
    fn i32(&mut self, x: i32) {
        self.bytes(&x.to_le_bytes());
    }
    fn u64(&mut self, x: u64) {
        self.bytes(&x.to_le_bytes());
    }
    fn i64(&mut self, x: i64) {
        self.bytes(&x.to_le_bytes());
    }
    fn len(&mut self, n: usize) {
        self.u32(n as u32);
    }
    fn str(&mut self, s: &str) {
        self.len(s.len());
        self.bytes(s.as_bytes());
    }
    fn span(&mut self, s: wrela_diag::Span) {
        self.u32(s.file.0);
        self.u32(s.start);
        self.u32(s.end);
    }
}

fn feed_scalar<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, s: Scalar) {
    let t = match s {
        Scalar::Bool => 0,
        Scalar::I8 => 1,
        Scalar::U8 => 2,
        Scalar::I16 => 3,
        Scalar::U16 => 4,
        Scalar::I32 => 5,
        Scalar::U32 => 6,
        Scalar::I64 => 7,
        Scalar::U64 => 8,
        Scalar::F32 => 9,
        Scalar::F64 => 10,
        Scalar::F16 => 11,
    };
    w.tag(t);
}

fn feed_type_def<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, d: &TypeDef) {
    match d {
        TypeDef::Scalar(s) => {
            w.tag(0);
            feed_scalar(w, *s);
        }
        TypeDef::Vector(s, n) => {
            w.tag(1);
            feed_scalar(w, *s);
            w.tag(*n);
        }
        TypeDef::Matrix(c, r) => {
            w.tag(2);
            w.tag(*c);
            w.tag(*r);
        }
        TypeDef::Struct { name, fields } => {
            w.tag(3);
            w.str(name);
            w.len(fields.len());
            for (n, t) in fields {
                w.str(n);
                w.u32(t.0);
            }
        }
        TypeDef::Enum { name, variants, tags } => {
            w.tag(4);
            w.str(name);
            w.len(variants.len());
            for (n, p) in variants {
                w.str(n);
                match p {
                    Some(t) => {
                        w.tag(1);
                        w.u32(t.0);
                    }
                    None => w.tag(0),
                }
            }
            w.len(tags.len());
            for &tag in tags {
                w.u32(tag);
            }
        }
        TypeDef::Array(e, n) => {
            w.tag(5);
            w.u32(e.0);
            w.u32(*n);
        }
        TypeDef::RuntimeArray(e) => {
            w.tag(6);
            w.u32(e.0);
        }
        TypeDef::Run(e) => {
            w.tag(7);
            w.u32(e.0);
        }
        TypeDef::Ptr(e) => {
            w.tag(8);
            w.u32(e.0);
        }
        TypeDef::Atomic(s) => {
            w.tag(9);
            feed_scalar(w, *s);
        }
    }
}

fn feed_tex_dim<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, d: TexDim) {
    let t = match d {
        TexDim::D2 => 0,
        TexDim::D2Array => 1,
        TexDim::Cube => 2,
        TexDim::D3 => 3,
    };
    w.tag(t);
}

fn feed_resource_kind<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, k: ResourceKind) {
    match k {
        ResourceKind::Uniform { storage } => {
            w.tag(0);
            w.bool(storage);
        }
        ResourceKind::StorageRead => w.tag(1),
        ResourceKind::StorageReadWrite => w.tag(2),
        ResourceKind::Private => w.tag(3),
        ResourceKind::Workgroup => w.tag(4),
        ResourceKind::Texture { depth, dim, format } => {
            w.tag(5);
            w.bool(depth);
            feed_tex_dim(w, dim);
            w.u32(format as u32);
        }
        ResourceKind::StorageTexture { dim, format } => {
            w.tag(6);
            feed_tex_dim(w, dim);
            w.u32(format as u32);
        }
        ResourceKind::Sampler { comparison } => {
            w.tag(7);
            w.bool(comparison);
        }
    }
}

fn feed_builtin_input<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, b: BuiltinInput) {
    let t = match b {
        BuiltinInput::GlobalInvocationId => 0,
        BuiltinInput::LocalInvocationId => 1,
        BuiltinInput::WorkgroupId => 2,
        BuiltinInput::NumWorkgroups => 3,
        BuiltinInput::VertexIndex => 4,
        BuiltinInput::InstanceIndex => 5,
        BuiltinInput::PrimitiveIndex => 6,
        BuiltinInput::Position => 7,
        BuiltinInput::SubgroupInvocationId => 8,
        BuiltinInput::SubgroupSize => 9,
    };
    w.tag(t);
}

fn feed_varying<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, v: &Varying) {
    w.len(v.path.len());
    for &p in &v.path {
        w.u32(p);
    }
    w.bool(v.position);
    w.bool(v.flat);
}

fn feed_stage<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, s: &Stage) {
    match s {
        Stage::Compute { workgroup_size } => {
            w.tag(0);
            for &x in workgroup_size {
                w.u32(x);
            }
        }
        Stage::Vertex { varyings } => {
            w.tag(1);
            w.len(varyings.len());
            for v in varyings {
                feed_varying(w, v);
            }
        }
        Stage::Fragment { varyings, outputs, depth, promise } => {
            w.tag(2);
            match varyings {
                Some((ty, vs)) => {
                    w.tag(1);
                    w.u32(ty.0);
                    w.len(vs.len());
                    for v in vs {
                        feed_varying(w, v);
                    }
                }
                None => w.tag(0),
            }
            w.len(outputs.len());
            for o in outputs {
                w.len(o.path.len());
                for &p in &o.path {
                    w.u32(p);
                }
                feed_scalar(w, o.scalar);
                w.tag(o.components);
            }
            match depth {
                Some(path) => {
                    w.tag(1);
                    w.len(path.len());
                    for &p in path {
                        w.u32(p);
                    }
                }
                None => w.tag(0),
            }
            match promise {
                None => w.tag(0),
                Some(DepthPromise::Greater) => w.tag(1),
                Some(DepthPromise::Less) => w.tag(2),
            }
        }
    }
}

fn feed_const<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, c: &Const) {
    match c {
        Const::Bool(b) => {
            w.tag(0);
            w.bool(*b);
        }
        Const::I32(x) => {
            w.tag(1);
            w.i32(*x);
        }
        Const::U32(x) => {
            w.tag(2);
            w.u32(*x);
        }
        Const::I64(x) => {
            w.tag(3);
            w.i64(*x);
        }
        Const::U64(x) => {
            w.tag(4);
            w.u64(*x);
        }
        Const::Small(s, x) => {
            w.tag(5);
            feed_scalar(w, *s);
            w.i64(*x);
        }
        Const::F32(x) => {
            w.tag(6);
            w.u32(x.to_bits());
        }
        Const::F64(x) => {
            w.tag(7);
            w.u64(x.to_bits());
        }
        Const::F16(x) => {
            w.tag(8);
            w.u32(x.to_bits());
        }
    }
}

fn feed_const_value<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, v: &ConstValue) {
    match v {
        ConstValue::Scalar(c) => {
            w.tag(0);
            feed_const(w, c);
        }
        ConstValue::Parts(ps) => {
            w.tag(1);
            w.len(ps.len());
            for p in ps {
                feed_const_value(w, p);
            }
        }
        ConstValue::Variant(tag, payload) => {
            w.tag(2);
            w.u32(*tag);
            match payload {
                Some(p) => {
                    w.tag(1);
                    feed_const_value(w, p);
                }
                None => w.tag(0),
            }
        }
        ConstValue::Bytes(bs) => {
            w.tag(3);
            w.len(bs.len());
            w.bytes(bs);
        }
        ConstValue::Addr(d) => {
            w.tag(4);
            w.u32(d.0);
        }
    }
}

fn feed_place<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, p: &Place) {
    match p.root {
        PlaceRoot::Local(l) => {
            w.tag(0);
            w.u32(l.0);
        }
        PlaceRoot::Param(i) => {
            w.tag(1);
            w.u32(i);
        }
        PlaceRoot::Resource(r) => {
            w.tag(2);
            w.u32(r.0);
        }
        PlaceRoot::Ptr(v) => {
            w.tag(3);
            w.u32(v.0);
        }
        PlaceRoot::Data(d) => {
            w.tag(4);
            w.u32(d.0);
        }
    }
    w.len(p.path.len());
    for proj in &p.path {
        match proj {
            Proj::Field(k) => {
                w.tag(0);
                w.u32(*k);
            }
            Proj::Comp(c) => {
                w.tag(1);
                w.tag(*c);
            }
            Proj::Index(v) => {
                w.tag(2);
                w.u32(v.0);
            }
        }
    }
}

fn feed_unop<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, op: UnOp) {
    w.tag(match op {
        UnOp::Neg => 0,
        UnOp::Not => 1,
    });
}

fn feed_binop<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, op: BinOp) {
    w.tag(match op {
        BinOp::Add => 0,
        BinOp::Sub => 1,
        BinOp::Mul => 2,
        BinOp::Div => 3,
        BinOp::Rem => 4,
        BinOp::WrappingAdd => 5,
        BinOp::WrappingSub => 6,
        BinOp::WrappingMul => 7,
        BinOp::Shl => 8,
        BinOp::Shr => 9,
        BinOp::BitAnd => 10,
        BinOp::BitOr => 11,
        BinOp::BitXor => 12,
        BinOp::And => 13,
        BinOp::Or => 14,
        BinOp::Eq => 15,
        BinOp::Ne => 16,
        BinOp::Lt => 17,
        BinOp::Le => 18,
        BinOp::Gt => 19,
        BinOp::Ge => 20,
    });
}

fn feed_builtin<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, b: Builtin) {
    w.tag(match b {
        Builtin::Sqrt => 0,
        Builtin::InverseSqrt => 1,
        Builtin::Sin => 2,
        Builtin::Cos => 3,
        Builtin::Tan => 4,
        Builtin::Asin => 5,
        Builtin::Acos => 6,
        Builtin::Atan => 7,
        Builtin::Atan2 => 8,
        Builtin::Exp => 9,
        Builtin::Exp2 => 10,
        Builtin::Log => 11,
        Builtin::Log2 => 12,
        Builtin::Sinh => 13,
        Builtin::Cosh => 14,
        Builtin::Tanh => 15,
        Builtin::Pow => 16,
        Builtin::Floor => 17,
        Builtin::Ceil => 18,
        Builtin::Round => 19,
        Builtin::Trunc => 20,
        Builtin::Fract => 21,
        Builtin::Abs => 22,
        Builtin::Sign => 23,
        Builtin::Min => 24,
        Builtin::Max => 25,
        Builtin::Clamp => 26,
        Builtin::Saturate => 27,
        Builtin::Mix => 28,
        Builtin::Step => 29,
        Builtin::Smoothstep => 30,
        Builtin::Length => 31,
        Builtin::Distance => 32,
        Builtin::Dot => 33,
        Builtin::Cross => 34,
        Builtin::Normalize => 35,
        Builtin::Dpdx => 36,
        Builtin::Dpdy => 37,
        Builtin::Fwidth => 38,
        Builtin::AllEqual => 39,
        Builtin::CountOnes => 40,
        Builtin::LeadingZeros => 41,
        Builtin::TrailingZeros => 42,
    });
}

fn feed_texture_op<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, op: TextureOp) {
    match op {
        TextureOp::Sample => w.tag(0),
        TextureOp::SampleLevel => w.tag(1),
        TextureOp::SampleGrad => w.tag(2),
        TextureOp::SampleBias => w.tag(3),
        TextureOp::SampleCompare => w.tag(4),
        TextureOp::SampleCompareLevel => w.tag(5),
        TextureOp::Gather(c) => {
            w.tag(6);
            w.tag(c);
        }
        TextureOp::GatherCompare => w.tag(7),
        TextureOp::Load => w.tag(8),
        TextureOp::Width => w.tag(9),
        TextureOp::Height => w.tag(10),
        TextureOp::Depth => w.tag(11),
        TextureOp::Layers => w.tag(12),
        TextureOp::Levels => w.tag(13),
    }
}

fn feed_atomic_op<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, op: AtomicOp) {
    w.tag(match op {
        AtomicOp::Load => 0,
        AtomicOp::Store => 1,
        AtomicOp::Add => 2,
        AtomicOp::Sub => 3,
        AtomicOp::Min => 4,
        AtomicOp::Max => 5,
        AtomicOp::And => 6,
        AtomicOp::Or => 7,
        AtomicOp::Xor => 8,
        AtomicOp::Exchange => 9,
        AtomicOp::CompareExchange => 10,
    });
}

fn feed_subgroup_op<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, op: SubgroupOp) {
    w.tag(match op {
        SubgroupOp::Add => 0,
        SubgroupOp::ExclusiveAdd => 1,
        SubgroupOp::InclusiveAdd => 2,
        SubgroupOp::Min => 3,
        SubgroupOp::Max => 4,
        SubgroupOp::And => 5,
        SubgroupOp::Or => 6,
        SubgroupOp::Xor => 7,
        SubgroupOp::Any => 8,
        SubgroupOp::All => 9,
        SubgroupOp::Ballot => 10,
        SubgroupOp::BroadcastFirst => 11,
        SubgroupOp::Shuffle => 12,
    });
}

fn feed_expr<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, e: &Expr) {
    match e {
        Expr::Const(c) => {
            w.tag(0);
            feed_const(w, c);
        }
        Expr::Zero(t) => {
            w.tag(1);
            w.u32(t.0);
        }
        Expr::Load(p) => {
            w.tag(2);
            feed_place(w, p);
        }
        Expr::Unary(op, v) => {
            w.tag(3);
            feed_unop(w, *op);
            w.u32(v.0);
        }
        Expr::Binary(op, a, b) => {
            w.tag(4);
            feed_binop(w, *op);
            w.u32(a.0);
            w.u32(b.0);
        }
        Expr::Call(f, args) => {
            w.tag(5);
            w.u32(f.0);
            w.len(args.len());
            for a in args {
                match a {
                    Arg::Value(v) => {
                        w.tag(0);
                        w.u32(v.0);
                    }
                    Arg::Place(p) => {
                        w.tag(1);
                        feed_place(w, p);
                    }
                }
            }
        }
        Expr::Builtin(b, xs) => {
            w.tag(6);
            feed_builtin(w, *b);
            w.len(xs.len());
            for x in xs {
                w.u32(x.0);
            }
        }
        Expr::Construct(t, xs) => {
            w.tag(7);
            w.u32(t.0);
            w.len(xs.len());
            for x in xs {
                w.u32(x.0);
            }
        }
        Expr::Variant(t, k, p) => {
            w.tag(8);
            w.u32(t.0);
            w.u32(*k);
            match p {
                Some(v) => {
                    w.tag(1);
                    w.u32(v.0);
                }
                None => w.tag(0),
            }
        }
        Expr::Extract(v, k) => {
            w.tag(9);
            w.u32(v.0);
            w.u32(*k);
        }
        Expr::ExtractDyn(v, i) => {
            w.tag(10);
            w.u32(v.0);
            w.u32(i.0);
        }
        Expr::Splat(v, n) => {
            w.tag(11);
            w.u32(v.0);
            w.tag(*n);
        }
        Expr::Swizzle(v, cs) => {
            w.tag(12);
            w.u32(v.0);
            w.len(cs.len());
            w.bytes(cs);
        }
        Expr::Convert(v, s) => {
            w.tag(13);
            w.u32(v.0);
            feed_scalar(w, *s);
        }
        Expr::Bitcast(v, s) => {
            w.tag(14);
            w.u32(v.0);
            feed_scalar(w, *s);
        }
        Expr::Select { cond, if_true, if_false } => {
            w.tag(15);
            w.u32(cond.0);
            w.u32(if_true.0);
            w.u32(if_false.0);
        }
        Expr::Run(p) => {
            w.tag(16);
            feed_place(w, p);
        }
        Expr::Addr(p) => {
            w.tag(17);
            feed_place(w, p);
        }
        Expr::Host(op, xs) => {
            w.tag(18);
            w.str(&format!("{op:?}"));
            w.len(xs.len());
            for x in xs {
                w.u32(x.0);
            }
        }
        Expr::Mem(op, xs) => {
            w.tag(19);
            w.str(&format!("{op:?}"));
            w.len(xs.len());
            for x in xs {
                w.u32(x.0);
            }
        }
        Expr::ArrayLength(p) => {
            w.tag(20);
            feed_place(w, p);
        }
        Expr::Texture(op, tex, samp, xs) => {
            w.tag(21);
            feed_texture_op(w, *op);
            w.u32(tex.0);
            match samp {
                Some(s) => {
                    w.tag(1);
                    w.u32(s.0);
                }
                None => w.tag(0),
            }
            w.len(xs.len());
            for x in xs {
                w.u32(x.0);
            }
        }
        Expr::TextureStore(tex, xs) => {
            w.tag(22);
            w.u32(tex.0);
            w.len(xs.len());
            for x in xs {
                w.u32(x.0);
            }
        }
        Expr::Atomic(op, p, xs) => {
            w.tag(23);
            feed_atomic_op(w, *op);
            feed_place(w, p);
            w.len(xs.len());
            for x in xs {
                w.u32(x.0);
            }
        }
        Expr::Subgroup(op, xs) => {
            w.tag(24);
            feed_subgroup_op(w, *op);
            w.len(xs.len());
            for x in xs {
                w.u32(x.0);
            }
        }
        Expr::Barrier => w.tag(25),
        Expr::Discard => w.tag(26),
        Expr::EntryInput(i) => {
            w.tag(27);
            w.u32(*i);
        }
        Expr::Param(i) => {
            w.tag(28);
            w.u32(*i);
        }
    }
}

fn feed_block<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, b: &[Stmt]) {
    w.len(b.len());
    for s in b {
        match s {
            Stmt::Let(v, e) => {
                w.tag(0);
                w.u32(v.0);
                feed_expr(w, e);
            }
            Stmt::Eval(e) => {
                w.tag(1);
                feed_expr(w, e);
            }
            Stmt::Store(p, v) => {
                w.tag(2);
                feed_place(w, p);
                w.u32(v.0);
            }
            Stmt::If { cond, then, else_ } => {
                w.tag(3);
                w.u32(cond.0);
                feed_block(w, then);
                feed_block(w, else_);
            }
            Stmt::Loop { body, continuing } => {
                w.tag(4);
                feed_block(w, body);
                feed_block(w, continuing);
            }
            Stmt::Break => w.tag(5),
            Stmt::Continue => w.tag(6),
            Stmt::Return(v) => {
                w.tag(7);
                match v {
                    Some(v) => {
                        w.tag(1);
                        w.u32(v.0);
                    }
                    None => w.tag(0),
                }
            }
            Stmt::Trap => w.tag(8),
            Stmt::At(span) => {
                w.tag(9);
                w.span(*span);
            }
        }
    }
}

fn feed_function<F: FnMut(&[u8])>(w: &mut Sink<'_, F>, f: &Function) {
    w.str(&f.name);
    w.len(f.params.len());
    for p in &f.params {
        w.str(&p.name);
        w.u32(p.ty.0);
        w.bool(p.by_ref);
        w.bool(p.mutable);
    }
    match f.ret {
        Some(t) => {
            w.tag(1);
            w.u32(t.0);
        }
        None => w.tag(0),
    }
    w.bool(f.ret_ref);
    w.len(f.locals.len());
    for l in &f.locals {
        w.str(&l.name);
        w.u32(l.ty.0);
    }
    w.len(f.values.len());
    for v in &f.values {
        w.u32(v.0);
    }
    feed_block(w, &f.body);
    w.bool(f.counted);
    w.bool(f.interval);
    match f.known {
        None => w.tag(0),
        Some(Known::Smin) => w.tag(1),
        Some(Known::Meet) => w.tag(2),
        Some(Known::ValueNoise) => w.tag(3),
        Some(Known::ValueNoiseBox) => w.tag(4),
        Some(Known::LatticeCoord) => w.tag(5),
        Some(Known::Pcg) => w.tag(6),
    }
    w.bool(f.relaxable);
    match f.budget {
        None => w.tag(0),
        Some(Budget { work, weight, at }) => {
            w.tag(1);
            match work {
                Some(x) => {
                    w.tag(1);
                    w.u64(x);
                }
                None => w.tag(0),
            }
            match weight {
                Some(x) => {
                    w.tag(1);
                    w.u64(x);
                }
                None => w.tag(0),
            }
            w.span(at);
        }
    }
}
