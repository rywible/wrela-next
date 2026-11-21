//! A GPU module built in Rust: what a shader the compiler doesn't compile from wrela is written
//! as (the host's own kernels, a host test's shaders), then written for its target by the
//! compiler's own writers ([`crate::emit_shader`]). One description, both targets.
//!
//! ```text
//! let mut b = Builder::new();
//! let u32t = b.scalar(Scalar::U32);
//! let out = b.storage("out", 1, u32t, true);
//! b.compute("store", [1, 1, 1], &[], |f| {
//!     let v = f.u32(42);
//!     let zero = f.u32(0);
//!     f.store(Builder::element(out, zero), v);
//! });
//! let shader = b.write(&mut pipeline, Target::HERE)?;
//! ```

use crate::{Checks, Shader};
use wrela_ir as ir;
pub use wrela_ir::{
    AtomicOp, BinOp, Builtin, BuiltinInput, Expr, Place, PlaceRoot, Proj, ResourceId, Scalar,
    TexDim, TextureFormat, TypeId, ValueId,
};

/// Builds a GPU module: its types, resources and entry points.
#[derive(Default)]
pub struct Builder {
    pub m: ir::Module,
}

impl Builder {
    pub fn new() -> Builder {
        Builder::default()
    }

    pub fn scalar(&mut self, s: Scalar) -> TypeId {
        self.m.types.scalar(s)
    }

    pub fn vector(&mut self, s: Scalar, n: u8) -> TypeId {
        self.m.types.vector_of(s, n)
    }

    pub fn array(&mut self, elem: TypeId, n: u32) -> TypeId {
        self.m.types.intern(ir::TypeDef::Array(elem, n))
    }

    pub fn atomic(&mut self, s: Scalar) -> TypeId {
        self.m.types.intern(ir::TypeDef::Atomic(s))
    }

    pub fn structure(&mut self, name: &str, fields: &[(&str, TypeId)]) -> TypeId {
        let fields = fields.iter().map(|(n, t)| (n.to_string(), *t)).collect();
        self.m.types.intern(ir::TypeDef::Struct { name: name.into(), fields })
    }

    fn resource(
        &mut self,
        name: &str,
        binding: u32,
        kind: ir::ResourceKind,
        ty: TypeId,
    ) -> ResourceId {
        self.m.add_resource(ir::Resource { name: name.into(), binding, kind, ty })
    }

    /// A uniform block of type `ty` (a struct) at `binding`.
    pub fn uniform(&mut self, name: &str, binding: u32, ty: TypeId) -> ResourceId {
        self.resource(name, binding, ir::ResourceKind::Uniform { storage: false }, ty)
    }

    /// A storage buffer of `elem`s at `binding`, written if `writable`.
    pub fn storage(
        &mut self,
        name: &str,
        binding: u32,
        elem: TypeId,
        writable: bool,
    ) -> ResourceId {
        self.m.types.intern(ir::TypeDef::RuntimeArray(elem));
        let kind = if writable {
            ir::ResourceKind::StorageReadWrite
        } else {
            ir::ResourceKind::StorageRead
        };
        self.resource(name, binding, kind, elem)
    }

    /// A texture at `binding`: its dimension, its format, and whether it holds depths.
    pub fn texture(
        &mut self,
        name: &str,
        binding: u32,
        dim: TexDim,
        format: TextureFormat,
        depth: bool,
    ) -> ResourceId {
        let ty = self.scalar(Scalar::F32);
        self.resource(name, binding, ir::ResourceKind::Texture { depth, dim, format }, ty)
    }

    /// Workgroup memory of type `ty`.
    pub fn workgroup(&mut self, name: &str, ty: TypeId) -> ResourceId {
        self.resource(name, 0, ir::ResourceKind::Workgroup, ty)
    }

    /// The place that is all of resource `r`.
    pub fn whole(r: ResourceId) -> Place {
        Place::root(PlaceRoot::Resource(r))
    }

    /// Element `i` of storage buffer `r`.
    pub fn element(r: ResourceId, i: ValueId) -> Place {
        Place::root(PlaceRoot::Resource(r)).with(Proj::Index(i))
    }

    /// A function that stays a call: its parameters' types, its result, and its body.
    pub fn function(
        &mut self,
        name: &str,
        params: &[TypeId],
        ret: Option<TypeId>,
        body: impl FnOnce(&mut Body<'_>),
    ) -> ir::FuncId {
        let params =
            params.iter().enumerate().map(|(k, t)| ir::Param::value(format!("p{k}"), *t)).collect();
        let f = ir::Function::new(name, params, ret);
        self.with_body(f, body)
    }

    fn with_body(&mut self, f: ir::Function, body: impl FnOnce(&mut Body<'_>)) -> ir::FuncId {
        let mut b = Body { m: &mut self.m, f, blocks: vec![Vec::new()] };
        body(&mut b);
        let Body { mut f, mut blocks, .. } = b;
        f.body = blocks.pop().unwrap_or_default();
        self.m.add_function(f)
    }

    /// A kernel of `workgroup_size` invocations, reading the builtin `inputs` (`Body::input`).
    pub fn compute(
        &mut self,
        name: &str,
        workgroup_size: [u32; 3],
        inputs: &[BuiltinInput],
        body: impl FnOnce(&mut Body<'_>),
    ) {
        let function = self.with_body(ir::Function::new(name, Vec::new(), None), body);
        self.m.entry_points.push(ir::EntryPoint {
            name: name.into(),
            stage: ir::Stage::Compute { workgroup_size },
            function,
            inputs: inputs.to_vec(),
        });
    }

    /// A vertex shader returning a clip position (a `vec4<f32>`), reading the builtin `inputs`.
    pub fn vertex(
        &mut self,
        name: &str,
        inputs: &[BuiltinInput],
        body: impl FnOnce(&mut Body<'_>),
    ) {
        let clip = self.clip_position();
        let function = self.with_body(ir::Function::new(name, Vec::new(), Some(clip)), body);
        self.m.entry_points.push(ir::EntryPoint {
            name: name.into(),
            stage: ir::Stage::Vertex {
                varyings: vec![ir::Varying { path: vec![0], position: true, flat: false }],
            },
            function,
            inputs: inputs.to_vec(),
        });
    }

    /// The struct a vertex shader returns: its clip position (`vertex` returns one).
    fn clip_position(&mut self) -> TypeId {
        let v4 = self.vector(Scalar::F32, 4);
        self.structure("ClipPosition", &[("position", v4)])
    }

    /// A fragment shader returning one colour (a `vec4<f32>`), reading the builtin `inputs`.
    pub fn fragment(
        &mut self,
        name: &str,
        inputs: &[BuiltinInput],
        body: impl FnOnce(&mut Body<'_>),
    ) {
        let v4 = self.vector(Scalar::F32, 4);
        let function = self.with_body(ir::Function::new(name, Vec::new(), Some(v4)), body);
        self.m.entry_points.push(ir::EntryPoint {
            name: name.into(),
            stage: ir::Stage::Fragment {
                varyings: None,
                outputs: vec![ir::FragmentColour {
                    path: Vec::new(),
                    scalar: Scalar::F32,
                    components: 4,
                }],
                depth: None,
                promise: None,
            },
            function,
            inputs: inputs.to_vec(),
        });
    }

    /// The module written for pipeline `p` on `target`, checked as the compiler's are
    /// ([`Checks::CHECKED`]); `p`'s sizes binding and entry points' names set from it.
    pub fn write(
        &self,
        p: &mut wrela_abi::manifest::Pipeline,
        target: wrela_abi::Target,
    ) -> Result<Shader, String> {
        self.write_with(p, target, Checks::CHECKED)
    }

    /// [`Builder::write`] with `checks` (exact mode's, say).
    pub fn write_with(
        &self,
        p: &mut wrela_abi::manifest::Pipeline,
        target: wrela_abi::Target,
        checks: Checks,
    ) -> Result<Shader, String> {
        let out = crate::emit_shader(&self.m, p, target, checks)?;
        use wrela_abi::manifest::Stage;
        match &mut p.stage {
            Stage::Compute { entry, .. } => *entry = out.entry_points[0].clone(),
            Stage::Render { vertex_entry, fragment_entry, .. } => {
                // The module's vertex shader and fragment shader, whatever the manifest named
                // them.
                let of = |vertex: bool| {
                    self.m.entry_points.iter().position(|e| match e.stage {
                        ir::Stage::Vertex { .. } => vertex,
                        ir::Stage::Fragment { .. } => !vertex,
                        ir::Stage::Compute { .. } => false,
                    })
                };
                if let Some(k) = of(true) {
                    *vertex_entry = out.entry_points[k].clone();
                }
                if let Some(k) = of(false) {
                    *fragment_entry = out.entry_points[k].clone();
                }
            }
        }
        Ok(out)
    }
}

/// A function's body being built: each method adds a statement, and returns the value it
/// defines.
pub struct Body<'a> {
    m: &'a mut ir::Module,
    f: ir::Function,
    /// The blocks being written, innermost last.
    blocks: Vec<ir::Block>,
}

impl Body<'_> {
    fn push(&mut self, s: ir::Stmt) {
        self.blocks.last_mut().expect("a block").push(s);
    }

    /// A value of type `ty` defined as `e`.
    pub fn let_(&mut self, ty: TypeId, e: Expr) -> ValueId {
        let v = self.f.new_value(ty);
        self.push(ir::Stmt::Let(v, e));
        v
    }

    pub fn ty(&self, v: ValueId) -> TypeId {
        self.f.value_ty(v)
    }

    pub fn types(&mut self) -> &mut ir::Types {
        &mut self.m.types
    }

    pub fn u32(&mut self, x: u32) -> ValueId {
        let t = self.m.types.u32();
        self.let_(t, Expr::Const(ir::Const::U32(x)))
    }

    pub fn i32(&mut self, x: i32) -> ValueId {
        let t = self.m.types.scalar(Scalar::I32);
        self.let_(t, Expr::Const(ir::Const::I32(x)))
    }

    pub fn f32(&mut self, x: f32) -> ValueId {
        let t = self.m.types.f32();
        self.let_(t, Expr::Const(ir::Const::F32(x)))
    }

    pub fn bool(&mut self, b: bool) -> ValueId {
        let t = self.m.types.bool();
        self.let_(t, Expr::Const(ir::Const::Bool(b)))
    }

    /// Builtin input `i` of the entry point (by its place in the entry's inputs), as the type
    /// `ty` holds it: a struct of its parts, or its one value.
    pub fn input(&mut self, i: u32, ty: TypeId) -> ValueId {
        self.let_(ty, Expr::EntryInput(i))
    }

    pub fn param(&mut self, i: u32) -> ValueId {
        let t = self.f.params[i as usize].ty;
        self.let_(t, Expr::Param(i))
    }

    /// What's at `p`, of type `ty`.
    pub fn load(&mut self, ty: TypeId, p: Place) -> ValueId {
        self.let_(ty, Expr::Load(p))
    }

    pub fn store(&mut self, p: Place, v: ValueId) {
        self.push(ir::Stmt::Store(p, v));
    }

    /// `a op b`: of `a`'s type, or a `bool` (of `a`'s width) for a comparison.
    pub fn binary(&mut self, op: BinOp, a: ValueId, b: ValueId) -> ValueId {
        let ta = self.ty(a);
        let ty = if op.is_comparison() {
            match self.m.types.get(ta) {
                ir::TypeDef::Vector(_, n) => {
                    let n = *n;
                    self.m.types.vector_of(Scalar::Bool, n)
                }
                _ => self.m.types.bool(),
            }
        } else {
            ta
        };
        self.let_(ty, Expr::Binary(op, a, b))
    }

    /// Builtin `b` of `args`, of type `ty`.
    pub fn builtin(&mut self, ty: TypeId, b: Builtin, args: &[ValueId]) -> ValueId {
        self.let_(ty, Expr::Builtin(b, args.to_vec()))
    }

    pub fn construct(&mut self, ty: TypeId, parts: &[ValueId]) -> ValueId {
        self.let_(ty, Expr::Construct(ty, parts.to_vec()))
    }

    /// Part `k` of `v` (a field, a component, a column), of type `ty`.
    pub fn extract(&mut self, ty: TypeId, v: ValueId, k: u32) -> ValueId {
        self.let_(ty, Expr::Extract(v, k))
    }

    /// Element `i` of `v` (an array or a vector value), of type `ty`.
    pub fn extract_dyn(&mut self, ty: TypeId, v: ValueId, i: ValueId) -> ValueId {
        self.let_(ty, Expr::ExtractDyn(v, i))
    }

    /// `v` made scalar `s` (a numeric conversion), of type `ty`.
    pub fn convert(&mut self, ty: TypeId, v: ValueId, s: Scalar) -> ValueId {
        self.let_(ty, Expr::Convert(v, s))
    }

    pub fn bitcast(&mut self, ty: TypeId, v: ValueId, s: Scalar) -> ValueId {
        self.let_(ty, Expr::Bitcast(v, s))
    }

    pub fn select(&mut self, cond: ValueId, if_true: ValueId, if_false: ValueId) -> ValueId {
        let ty = self.ty(if_true);
        self.let_(ty, Expr::Select { cond, if_true, if_false })
    }

    pub fn call(&mut self, f: ir::FuncId, args: &[ValueId]) -> Option<ValueId> {
        let args: Vec<ir::Arg> = args.iter().map(|v| ir::Arg::Value(*v)).collect();
        match self.m.functions[f.index()].ret {
            Some(t) => Some(self.let_(t, Expr::Call(f, args))),
            None => {
                self.push(ir::Stmt::Eval(Expr::Call(f, args)));
                None
            }
        }
    }

    /// An atomic operation on `p`: its old value, of type `ty`, or (a store) nothing.
    pub fn atomic(
        &mut self,
        ty: TypeId,
        op: AtomicOp,
        p: Place,
        args: &[ValueId],
    ) -> Option<ValueId> {
        if op == AtomicOp::Store {
            self.push(ir::Stmt::Eval(Expr::Atomic(op, p, args.to_vec())));
            return None;
        }
        Some(self.let_(ty, Expr::Atomic(op, p, args.to_vec())))
    }

    /// The texel of texture `t` at `x` and `y` (`u32`s), level `level`: a `vec4<f32>`.
    pub fn texel(&mut self, t: ResourceId, x: ValueId, y: ValueId, level: ValueId) -> ValueId {
        let v4 = self.m.types.vector(4);
        self.let_(v4, Expr::Texture(ir::TextureOp::Load, t, None, vec![x, y, level]))
    }

    /// `if cond { then } else { else_ }`.
    pub fn if_(
        &mut self,
        cond: ValueId,
        then: impl FnOnce(&mut Self),
        else_: impl FnOnce(&mut Self),
    ) {
        self.blocks.push(Vec::new());
        then(self);
        let then = self.blocks.pop().unwrap_or_default();
        self.blocks.push(Vec::new());
        else_(self);
        let else_ = self.blocks.pop().unwrap_or_default();
        self.push(ir::Stmt::If { cond, then, else_ });
    }

    /// `loop { body }`, left by `ret` or a break (`break_`).
    pub fn loop_(&mut self, body: impl FnOnce(&mut Self)) {
        self.blocks.push(Vec::new());
        body(self);
        let body = self.blocks.pop().unwrap_or_default();
        self.push(ir::Stmt::Loop { body, continuing: Vec::new() });
    }

    pub fn break_(&mut self) {
        self.push(ir::Stmt::Break);
    }

    /// A point valid code never reaches (the end of a loop left only by `return`).
    pub fn trap(&mut self) {
        self.push(ir::Stmt::Trap);
    }

    /// `workgroupBarrier()`.
    pub fn barrier(&mut self) {
        self.push(ir::Stmt::Eval(Expr::Barrier));
    }

    /// A local variable of type `ty`, zeroed: its place.
    pub fn local(&mut self, name: &str, ty: TypeId) -> Place {
        Place::local(self.f.new_local(name, ty))
    }

    pub fn ret(&mut self, v: Option<ValueId>) {
        self.push(ir::Stmt::Return(v));
    }

    /// Returns clip position `v` (a `vec4<f32>`) from a vertex shader built by
    /// [`Builder::vertex`].
    pub fn ret_clip(&mut self, v: ValueId) {
        let clip = self.f.ret.expect("a vertex shader's result");
        let c = self.construct(clip, &[v]);
        self.ret(Some(c));
    }

    /// The four bytes of `x` (a `u32`), lowest first, each over 255: WGSL's `unpack4x8unorm`.
    pub fn unpack4x8unorm(&mut self, x: ValueId) -> ValueId {
        let f = self.m.types.f32();
        let v4 = self.m.types.vector(4);
        let mask = self.u32(255);
        let scale = self.f32(255.0);
        let mut parts = Vec::new();
        for k in 0..4 {
            let shift = self.u32(8 * k);
            let s = self.binary(BinOp::Shr, x, shift);
            let b = self.binary(BinOp::BitAnd, s, mask);
            let bf = self.convert(f, b, Scalar::F32);
            parts.push(self.binary(BinOp::Div, bf, scale));
        }
        self.construct(v4, &parts)
    }
}
