//! The Vulkan back end, written from the IR directly: a GPU module of [`wrela_ir`] as SPIR-V 1.6
//! (Vulkan 1.3), each binding `DescriptorSet 0` at the number the host's layout gives it
//! ([`wrela_abi::vulkan::descriptor`]), for the host to hand Vulkan as it is.
//!
//! It computes what naga 30's SPIR-V writer computed for the same module (`spv`, which it
//! replaces), with the instructions it chose (`GLSL.std.450`'s `FMin`, `FClamp`, `FMix` with a
//! splatted selector, `RoundEven`, `OpFRem`, `OpVectorTimesScalar`, a float clamped before it's
//! made an integer, derivatives `Fine` as Metal's are, a composite read at a worked-out index
//! through a variable it's stored in), the same run-time checks ([`Checks`]: indexes clamped by
//! `UMin`, loops given a 64-bit bound, integer division guarded) and workgroup memory zeroed by
//! the first invocation before a barrier. naga's writer was read as reference
//! (docs/citations.md); nothing of it is copied. Beyond it:
//!
//! - **Explicit layout only where memory is shared.** A struct or array in a uniform or storage
//!   buffer is declared with its offsets and strides; the same type held in a variable of a
//!   function, an invocation or a workgroup is declared again without them (Vulkan forbids
//!   explicit layout there, VUID-StandaloneSpirv-None-10684), and `OpCopyLogical` moves a value
//!   between the two.
//! - **No `OpCompositeConstruct` of one constituent** (naga 30 wrote one for a whole vector, which
//!   SPIR-V forbids and RADV crashed on).
//! - A fragment shader's promised depth is `DepthGreater` or `DepthLess` beside
//!   `DepthReplacing`.
//!
//! Memory model `Logical GLSL450`, as wgpu's host enables no Vulkan memory model.

use crate::Checks;
use spirv::{
    BuiltIn, Capability, Decoration, Dim, ExecutionMode, ExecutionModel, GlslStd450Op as GLOp,
    ImageFormat, Op, StorageClass,
};
use std::collections::{BTreeSet, HashMap};
use wrela_abi::manifest::{Pipeline, UniformSpace};
use wrela_ir as ir;
use wrela_ir::layout::WgslLayouts;

type R<T> = Result<T, String>;
type Word = u32;

/// The SPIR-V version written: what a Vulkan 1.3 device takes (RADV on the Linux machine is 1.4).
pub const LANGUAGE: (u8, u8) = (1, 6);

/// Module `m` as SPIR-V's words in bytes (little-endian) for pipeline `p`, each of its bindings
/// at the Vulkan binding [`wrela_abi::vulkan::descriptor`] gives it, with `promise` on its
/// fragment shader's depth and `checks` written in; and its entry points' names.
pub fn write(
    m: &ir::Module,
    p: &Pipeline,
    promise: Option<ir::DepthPromise>,
    checks: Checks,
) -> R<(Vec<u8>, Vec<String>)> {
    let mut w = Writer::new(m, p, promise, checks);
    w.module()?;
    let words = w.finish();
    Ok((words.iter().flat_map(|w| w.to_le_bytes()).collect(), w.entry_names))
}

// ---- instructions ----------------------------------------------------------------------------------

/// One instruction's words, its opcode and word count first.
fn inst(op: Op, operands: &[Word]) -> Vec<Word> {
    let mut out = Vec::with_capacity(operands.len() + 1);
    out.push(((operands.len() as u32 + 1) << 16) | op as u32);
    out.extend_from_slice(operands);
    out
}

/// A string as SPIR-V's literal: its UTF-8 bytes, nul-terminated, in little-endian words.
fn string(s: &str) -> Vec<Word> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// What a type is, to declare it once.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum TypeKey {
    Void,
    Bool,
    Int(bool),
    Float(u32),
    Vector(Word, u32),
    Matrix(Word, u32),
    /// An array of `len` (a constant's id), with a stride when laid out.
    Array(Word, Word, Option<u32>),
    RuntimeArray(Word, u32),
    /// An array of descriptors (images or samplers) of the device's table: no stride.
    DescriptorArray(Word),
    /// An IR struct or enum, laid out or not.
    Struct(ir::TypeId, bool),
    /// A struct of its own: a buffer's block, or a compare-exchange's result.
    Wrapper(String),
    Pointer(StorageClass, Word),
    Function(Word, Vec<Word>),
    Image {
        sampled: Word,
        dim: Dim,
        depth: bool,
        arrayed: bool,
        storage: Option<ImageFormat>,
    },
    Sampler,
    SampledImage(Word),
}

/// A constant, to declare it once.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ConstKey {
    Scalar(Word, u32),
    Composite(Word, Vec<Word>),
    Null(Word),
    True(Word),
    False(Word),
}

/// What a value's id holds: whether it's a constant (so a composite of it is one too).
#[derive(Clone, Copy, Debug)]
struct Value {
    id: Word,
    constant: bool,
}

// ---- the module ------------------------------------------------------------------------------------

struct Writer<'m> {
    m: &'m ir::Module,
    p: &'m Pipeline,
    promise: Option<ir::DepthPromise>,
    checks: Checks,
    /// What the compiler proves of the checks: the ones it makes unneeded are left out.
    proofs: wrela_ir::work::Proofs,
    layouts: WgslLayouts,
    next: Word,
    capabilities: BTreeSet<u32>,
    glsl: Word,
    entry_points: Vec<Word>,
    execution_modes: Vec<Word>,
    debug: Vec<Word>,
    annotations: Vec<Word>,
    globals: Vec<Word>,
    functions: Vec<Word>,
    types: HashMap<TypeKey, Word>,
    constants: HashMap<ConstKey, Word>,
    /// Each IR type's id, plain and laid out.
    ir_types: HashMap<(ir::TypeId, bool), Word>,
    /// Each resource's variable, its storage class, and (a sized buffer's) its block's type.
    resources: Vec<Option<(Word, StorageClass)>>,
    /// Each function's id.
    function_ids: Vec<Word>,
    /// The resources each function uses, itself or through its calls.
    uses: Vec<BTreeSet<usize>>,
    entry_names: Vec<String>,
    /// The structs decorated `Block`.
    blocks: BTreeSet<Word>,
    /// The push constants' variable once declared (the block of indices), and the table's array
    /// variables by the type of image (or sampler) they hold.
    push: Option<Word>,
    tables: HashMap<Word, Word>,
}

impl<'m> Writer<'m> {
    fn new(
        m: &'m ir::Module,
        p: &'m Pipeline,
        promise: Option<ir::DepthPromise>,
        checks: Checks,
    ) -> Writer<'m> {
        let mut w = Writer {
            m,
            p,
            promise,
            checks,
            proofs: if checks == Checks::UNCHECKED {
                wrela_ir::work::Proofs::default()
            } else {
                wrela_ir::work::proofs(m)
            },
            layouts: WgslLayouts::default(),
            next: 1,
            capabilities: BTreeSet::new(),
            glsl: 0,
            entry_points: Vec::new(),
            execution_modes: Vec::new(),
            debug: Vec::new(),
            annotations: Vec::new(),
            globals: Vec::new(),
            functions: Vec::new(),
            types: HashMap::new(),
            constants: HashMap::new(),
            ir_types: HashMap::new(),
            resources: vec![None; m.resources.len()],
            function_ids: Vec::new(),
            uses: Vec::new(),
            entry_names: Vec::new(),
            blocks: BTreeSet::new(),
            push: None,
            tables: HashMap::new(),
        };
        w.glsl = w.id();
        w.capability(Capability::Shader);
        w.function_ids = (0..m.functions.len()).map(|_| w.id()).collect();
        w.uses = resource_uses(m);
        w
    }

    fn id(&mut self) -> Word {
        let id = self.next;
        self.next += 1;
        id
    }

    fn capability(&mut self, c: Capability) {
        self.capabilities.insert(c as u32);
    }

    fn name(&mut self, id: Word, name: &str) {
        let mut operands = vec![id];
        operands.extend(string(name));
        self.debug.extend(inst(Op::Name, &operands));
    }

    fn decorate(&mut self, id: Word, d: Decoration, extra: &[Word]) {
        let mut operands = vec![id, d as u32];
        operands.extend_from_slice(extra);
        self.annotations.extend(inst(Op::Decorate, &operands));
    }

    fn member_decorate(&mut self, id: Word, member: u32, d: Decoration, extra: &[Word]) {
        let mut operands = vec![id, member, d as u32];
        operands.extend_from_slice(extra);
        self.annotations.extend(inst(Op::MemberDecorate, &operands));
    }

    fn finish(&mut self) -> Vec<Word> {
        let mut out = vec![
            spirv::MAGIC_NUMBER,
            (u32::from(LANGUAGE.0) << 16) | (u32::from(LANGUAGE.1) << 8),
            0,
            self.next,
            0,
        ];
        for c in &self.capabilities {
            out.extend(inst(Op::Capability, &[*c]));
        }
        let mut import = vec![self.glsl];
        import.extend(string("GLSL.std.450"));
        out.extend(inst(Op::ExtInstImport, &import));
        out.extend(inst(
            Op::MemoryModel,
            &[spirv::AddressingModel::Logical as u32, spirv::MemoryModel::GLSL450 as u32],
        ));
        out.extend_from_slice(&self.entry_points);
        out.extend_from_slice(&self.execution_modes);
        out.extend_from_slice(&self.debug);
        out.extend_from_slice(&self.annotations);
        out.extend_from_slice(&self.globals);
        out.extend_from_slice(&self.functions);
        out
    }

    // ---- types ----

    fn type_id(&mut self, key: TypeKey) -> Word {
        if let Some(&id) = self.types.get(&key) {
            return id;
        }
        let id = self.id();
        let words = match &key {
            TypeKey::Void => inst(Op::TypeVoid, &[id]),
            TypeKey::Bool => inst(Op::TypeBool, &[id]),
            TypeKey::Int(signed) => inst(Op::TypeInt, &[id, 32, u32::from(*signed)]),
            TypeKey::Float(width) => {
                if *width == 16 {
                    self.capability(Capability::Float16);
                }
                inst(Op::TypeFloat, &[id, *width])
            }
            TypeKey::Vector(c, n) => inst(Op::TypeVector, &[id, *c, *n]),
            TypeKey::Matrix(c, n) => inst(Op::TypeMatrix, &[id, *c, *n]),
            TypeKey::Array(e, len, stride) => {
                if let Some(s) = stride {
                    self.decorate(id, Decoration::ArrayStride, &[*s]);
                }
                inst(Op::TypeArray, &[id, *e, *len])
            }
            TypeKey::DescriptorArray(e) => inst(Op::TypeRuntimeArray, &[id, *e]),
            TypeKey::RuntimeArray(e, stride) => {
                self.decorate(id, Decoration::ArrayStride, &[*stride]);
                inst(Op::TypeRuntimeArray, &[id, *e])
            }
            TypeKey::Struct(..) | TypeKey::Wrapper(_) => {
                unreachable!("structs are declared by their own functions")
            }
            TypeKey::Pointer(class, t) => inst(Op::TypePointer, &[id, *class as u32, *t]),
            TypeKey::Function(ret, params) => {
                let mut operands = vec![id, *ret];
                operands.extend_from_slice(params);
                inst(Op::TypeFunction, &operands)
            }
            TypeKey::Image { sampled, dim, depth, arrayed, storage } => inst(
                Op::TypeImage,
                &[
                    id,
                    *sampled,
                    *dim as u32,
                    u32::from(*depth),
                    u32::from(*arrayed),
                    0,
                    if storage.is_some() { 2 } else { 1 },
                    storage.unwrap_or(ImageFormat::Unknown) as u32,
                ],
            ),
            TypeKey::Sampler => inst(Op::TypeSampler, &[id]),
            TypeKey::SampledImage(i) => inst(Op::TypeSampledImage, &[id, *i]),
        };
        self.globals.extend(words);
        self.types.insert(key, id);
        id
    }

    fn void(&mut self) -> Word {
        self.type_id(TypeKey::Void)
    }

    fn bool_ty(&mut self) -> Word {
        self.type_id(TypeKey::Bool)
    }

    fn u32_ty(&mut self) -> Word {
        self.type_id(TypeKey::Int(false))
    }

    fn i32_ty(&mut self) -> Word {
        self.type_id(TypeKey::Int(true))
    }

    fn f32_ty(&mut self) -> Word {
        self.type_id(TypeKey::Float(32))
    }

    fn vec_ty(&mut self, scalar: Word, n: u32) -> Word {
        self.type_id(TypeKey::Vector(scalar, n))
    }

    fn pointer(&mut self, class: StorageClass, t: Word) -> Word {
        self.type_id(TypeKey::Pointer(class, t))
    }

    fn scalar_ty(&mut self, s: ir::Scalar) -> R<Word> {
        Ok(match s {
            ir::Scalar::Bool => self.bool_ty(),
            ir::Scalar::I32 => self.i32_ty(),
            ir::Scalar::U32 => self.u32_ty(),
            ir::Scalar::F32 => self.f32_ty(),
            ir::Scalar::F16 => self.type_id(TypeKey::Float(16)),
            other => return Err(format!("internal: `{}` in GPU code", other.name())),
        })
    }

    /// IR type `t`: `laid` with its offsets and strides (in a buffer), else without.
    fn ty(&mut self, t: ir::TypeId, laid: bool) -> R<Word> {
        let types = &self.m.types;
        // Only aggregates carry a layout.
        let laid = laid
            && matches!(
                types.get(t),
                ir::TypeDef::Array(..) | ir::TypeDef::Struct { .. } | ir::TypeDef::Enum { .. }
            );
        if let Some(&id) = self.ir_types.get(&(t, laid)) {
            return Ok(id);
        }
        let id = match types.get(t).clone() {
            ir::TypeDef::Scalar(s) | ir::TypeDef::Atomic(s) => self.scalar_ty(s)?,
            ir::TypeDef::Vector(s, n) => {
                let c = self.scalar_ty(s)?;
                self.vec_ty(c, u32::from(n))
            }
            ir::TypeDef::Matrix(c, r) => {
                let f = self.f32_ty();
                let col = self.vec_ty(f, u32::from(r));
                self.type_id(TypeKey::Matrix(col, u32::from(c)))
            }
            ir::TypeDef::Array(e, n) => {
                let elem = self.ty(e, laid)?;
                let len = self.const_u32(n);
                let stride = laid.then(|| self.layouts.get(&self.m.types, e).0.stride());
                self.type_id(TypeKey::Array(elem, len, stride))
            }
            ir::TypeDef::RuntimeArray(e) => {
                let elem = self.ty(e, true)?;
                let stride = self.layouts.get(&self.m.types, e).0.stride();
                self.type_id(TypeKey::RuntimeArray(elem, stride))
            }
            ir::TypeDef::Struct { ref name, .. } | ir::TypeDef::Enum { ref name, .. } => {
                let name = name.clone();
                self.struct_ty(t, laid, &name)?
            }
            ir::TypeDef::Run(_) | ir::TypeDef::Ptr(_) => {
                return Err("internal: a CPU-only type in GPU code".into());
            }
        };
        self.ir_types.insert((t, laid), id);
        Ok(id)
    }

    /// The members of a struct, or of an enum as the GPU holds it: its tag, then each payload.
    fn members(&self, t: ir::TypeId) -> R<Vec<(String, ir::TypeId)>> {
        let types = &self.m.types;
        match types.get(t) {
            ir::TypeDef::Struct { fields, .. } => {
                Ok(fields.iter().map(|(n, f)| (n.clone(), *f)).collect())
            }
            ir::TypeDef::Enum { variants, .. } => {
                let tag = types.field(t, 0).ok_or("internal: an enum's tag type")?;
                Ok(std::iter::once(("tag".to_string(), tag))
                    .chain(variants.iter().filter_map(|(v, p)| Some((v.clone(), (*p)?))))
                    .collect())
            }
            _ => Err("internal: members of a type that has none".into()),
        }
    }

    /// Struct `t`, its members' types first; laid out, with each member's offset (and a
    /// matrix's stride).
    fn struct_ty(&mut self, t: ir::TypeId, laid: bool, name: &str) -> R<Word> {
        let members = self.members(t)?;
        let mut ids = Vec::with_capacity(members.len());
        for (_, mt) in &members {
            ids.push(self.ty(*mt, laid)?);
        }
        let id = self.id();
        let mut operands = vec![id];
        operands.extend_from_slice(&ids);
        self.globals.extend(inst(Op::TypeStruct, &operands));
        self.name(id, &format!("{}{}", ir::ident(name), if laid { "_laid" } else { "" }));
        if laid {
            let offsets = self.layouts.get(&self.m.types, t).1.to_vec();
            for (k, (_, mt)) in members.iter().enumerate() {
                self.member_decorate(id, k as u32, Decoration::Offset, &[offsets[k]]);
                if let Some(rows) = matrix_rows(&self.m.types, *mt) {
                    self.member_decorate(id, k as u32, Decoration::ColMajor, &[]);
                    let stride = ir::layout::column_stride(rows);
                    self.member_decorate(id, k as u32, Decoration::MatrixStride, &[stride]);
                }
            }
        }
        self.types.insert(TypeKey::Struct(t, laid), id);
        Ok(id)
    }

    /// A struct of its own, named `name`, of `members`, with their offsets if given.
    fn wrapper(&mut self, name: &str, members: &[Word], offsets: Option<&[u32]>) -> Word {
        let key = TypeKey::Wrapper(name.to_string());
        if let Some(&id) = self.types.get(&key) {
            return id;
        }
        let id = self.id();
        let mut operands = vec![id];
        operands.extend_from_slice(members);
        self.globals.extend(inst(Op::TypeStruct, &operands));
        self.name(id, name);
        if let Some(offsets) = offsets {
            for (k, o) in offsets.iter().enumerate() {
                self.member_decorate(id, k as u32, Decoration::Offset, &[*o]);
            }
        }
        self.types.insert(key, id);
        id
    }

    // ---- constants ----

    fn constant(&mut self, key: ConstKey) -> Word {
        if let Some(&id) = self.constants.get(&key) {
            return id;
        }
        let id = self.id();
        let words = match &key {
            ConstKey::Scalar(t, bits) => inst(Op::Constant, &[*t, id, *bits]),
            ConstKey::Composite(t, parts) => {
                let mut operands = vec![*t, id];
                operands.extend_from_slice(parts);
                inst(Op::ConstantComposite, &operands)
            }
            ConstKey::Null(t) => inst(Op::ConstantNull, &[*t, id]),
            ConstKey::True(t) => inst(Op::ConstantTrue, &[*t, id]),
            ConstKey::False(t) => inst(Op::ConstantFalse, &[*t, id]),
        };
        self.globals.extend(words);
        self.constants.insert(key, id);
        id
    }

    fn const_u32(&mut self, x: u32) -> Word {
        let t = self.u32_ty();
        self.constant(ConstKey::Scalar(t, x))
    }

    fn const_i32(&mut self, x: i32) -> Word {
        let t = self.i32_ty();
        self.constant(ConstKey::Scalar(t, x as u32))
    }

    fn const_f32(&mut self, x: f32) -> Word {
        let t = self.f32_ty();
        self.constant(ConstKey::Scalar(t, x.to_bits()))
    }

    fn const_bool(&mut self, b: bool) -> Word {
        let t = self.bool_ty();
        if b { self.constant(ConstKey::True(t)) } else { self.constant(ConstKey::False(t)) }
    }

    fn null(&mut self, t: Word) -> Word {
        self.constant(ConstKey::Null(t))
    }

    /// `n` of constant `c` as a vector of type `t` (or `c` itself, for `n` = 1).
    fn splat_const(&mut self, t: Word, c: Word, n: u32) -> Word {
        if n <= 1 {
            return c;
        }
        self.constant(ConstKey::Composite(t, vec![c; n as usize]))
    }

    fn scalar_const(&mut self, c: &ir::Const) -> R<Word> {
        Ok(match c {
            ir::Const::Bool(b) => self.const_bool(*b),
            ir::Const::I32(x) => self.const_i32(*x),
            ir::Const::U32(x) => self.const_u32(*x),
            ir::Const::F32(x) => self.const_f32(*x),
            ir::Const::F16(x) => {
                let t = self.type_id(TypeKey::Float(16));
                let bits = u32::from(half::f16::from_f32(*x).to_bits());
                self.constant(ConstKey::Scalar(t, bits))
            }
            other => return Err(format!("internal: the constant {other:?} in GPU code")),
        })
    }

    /// Constant value `v` of IR type `t`, plain.
    fn const_value(&mut self, t: ir::TypeId, v: &ir::ConstValue) -> R<Word> {
        let ty = self.ty(t, false)?;
        match (self.m.types.get(t).clone(), v) {
            (ir::TypeDef::Scalar(_), ir::ConstValue::Scalar(c)) => self.scalar_const(c),
            (ir::TypeDef::Vector(..), ir::ConstValue::Parts(parts)) => {
                let mut ids = Vec::new();
                for p in parts {
                    let ir::ConstValue::Scalar(c) = p else {
                        return Err("internal: a vector's part in a table".into());
                    };
                    ids.push(self.scalar_const(c)?);
                }
                Ok(self.constant(ConstKey::Composite(ty, ids)))
            }
            (ir::TypeDef::Array(e, _), ir::ConstValue::Parts(parts)) => {
                let ids = parts.iter().map(|p| self.const_value(e, p)).collect::<R<Vec<_>>>()?;
                Ok(self.constant(ConstKey::Composite(ty, ids)))
            }
            (ir::TypeDef::Struct { fields, .. }, ir::ConstValue::Parts(parts)) => {
                let mut ids = Vec::new();
                for ((_, ft), p) in fields.iter().zip(parts) {
                    ids.push(self.const_value(*ft, p)?);
                }
                Ok(self.constant(ConstKey::Composite(ty, ids)))
            }
            _ => Err("internal: a table's value doesn't fit its type".into()),
        }
    }

    // ---- resources ----

    /// What resource `r`'s variable points at, and its storage class (a texture or sampler of
    /// the device's table: its element's type).
    fn pointee(&mut self, r: usize) -> R<(Word, StorageClass)> {
        let res = &self.m.resources[r];
        Ok(match res.kind {
            ir::ResourceKind::Uniform { storage } => {
                let struct_id = self.ty(res.ty, true)?;
                if self.blocks.insert(struct_id) {
                    self.decorate(struct_id, Decoration::Block, &[]);
                }
                let storage = storage
                    || self.p.uniform.as_ref().is_some_and(|u| u.space == UniformSpace::Storage);
                if storage {
                    (struct_id, StorageClass::StorageBuffer)
                } else {
                    (struct_id, StorageClass::Uniform)
                }
            }
            ir::ResourceKind::StorageRead | ir::ResourceKind::StorageReadWrite => {
                let rt = self.m.types.lookup(&ir::TypeDef::RuntimeArray(res.ty));
                let array = match rt {
                    Some(rt) => self.ty(rt, true)?,
                    None => {
                        let elem = self.ty(res.ty, true)?;
                        let stride = self.layouts.get(&self.m.types, res.ty).0.stride();
                        self.type_id(TypeKey::RuntimeArray(elem, stride))
                    }
                };
                let key = format!("block{array}");
                let new = !self.types.contains_key(&TypeKey::Wrapper(key.clone()));
                let block = self.wrapper(&key, &[array], Some(&[0]));
                if new {
                    self.decorate(block, Decoration::Block, &[]);
                    if let Some(rows) = matrix_rows(&self.m.types, res.ty) {
                        self.member_decorate(block, 0, Decoration::ColMajor, &[]);
                        let stride = ir::layout::column_stride(rows);
                        self.member_decorate(block, 0, Decoration::MatrixStride, &[stride]);
                    }
                }
                (block, StorageClass::StorageBuffer)
            }
            ir::ResourceKind::Private => (self.ty(res.ty, false)?, StorageClass::Private),
            ir::ResourceKind::Workgroup => (self.ty(res.ty, false)?, StorageClass::Workgroup),
            ir::ResourceKind::Texture { depth, dim, format } => {
                let sampled = if ir::texel_scalar(format) == ir::Scalar::U32 && !depth {
                    self.u32_ty()
                } else {
                    self.f32_ty()
                };
                let (d, arrayed) = image_dim(dim);
                (
                    self.type_id(TypeKey::Image { sampled, dim: d, depth, arrayed, storage: None }),
                    StorageClass::UniformConstant,
                )
            }
            ir::ResourceKind::StorageTexture { dim, format } => {
                let sampled = if ir::texel_scalar(format) == ir::Scalar::U32 {
                    self.u32_ty()
                } else {
                    self.f32_ty()
                };
                let storage = match format {
                    ir::TextureFormat::Rgba8 => ImageFormat::Rgba8,
                    ir::TextureFormat::Rgba16Float => ImageFormat::Rgba16f,
                    ir::TextureFormat::R16Float => ImageFormat::R16f,
                    ir::TextureFormat::R32Float => ImageFormat::R32f,
                    ir::TextureFormat::R32Uint => ImageFormat::R32ui,
                    ir::TextureFormat::Rg32Float => ImageFormat::Rg32f,
                    ir::TextureFormat::Rgba32Float => ImageFormat::Rgba32f,
                    ir::TextureFormat::Rg32Uint => ImageFormat::Rg32ui,
                    f => return Err(format!("internal: kernels can't write a {f:?} texture")),
                };
                // (Vulkan's `shaderStorageImageExtendedFormats`, which the device turns on.)
                if matches!(storage, ImageFormat::R16f | ImageFormat::Rg32f | ImageFormat::Rg32ui) {
                    self.capability(Capability::StorageImageExtendedFormats);
                }
                let (d, arrayed) = image_dim(dim);
                (
                    self.type_id(TypeKey::Image {
                        sampled,
                        dim: d,
                        depth: false,
                        arrayed,
                        storage: Some(storage),
                    }),
                    StorageClass::UniformConstant,
                )
            }
            ir::ResourceKind::Sampler { .. } => {
                (self.type_id(TypeKey::Sampler), StorageClass::UniformConstant)
            }
        })
    }

    /// Resource `r`'s variable and storage class, declared the first time.
    fn resource(&mut self, r: usize) -> R<(Word, StorageClass)> {
        if let Some(v) = self.resources[r] {
            return Ok(v);
        }
        let res = &self.m.resources[r];
        let res_name = res.name.clone();
        let binding = res.binding;
        let (pointee, class) = self.pointee(r)?;
        if wrela_abi::table::index_offset(self.p, binding).is_some() {
            // Its variable is the table's array of its type; a function reaches its element.
            let storage = matches!(res.kind, ir::ResourceKind::StorageTexture { .. });
            let sampler = matches!(res.kind, ir::ResourceKind::Sampler { .. });
            let var = self.table_var(pointee, storage, sampler);
            self.resources[r] = Some((var, class));
            return Ok((var, class));
        }
        let ptr = self.pointer(class, pointee);
        let var = self.id();
        let init = match class {
            StorageClass::Private => Some(self.null(pointee)),
            _ => None,
        };
        let mut operands = vec![ptr, var, class as u32];
        operands.extend(init);
        self.globals.extend(inst(Op::Variable, &operands));
        self.name(var, &ir::ident(&res_name));
        if !matches!(class, StorageClass::Private | StorageClass::Workgroup) {
            let descriptor = wrela_abi::vulkan::descriptor(self.p, binding)
                .ok_or_else(|| format!("internal: binding 0:{binding} isn't in the layout"))?;
            self.decorate(var, Decoration::DescriptorSet, &[0]);
            self.decorate(var, Decoration::Binding, &[descriptor]);
        }
        let read_only = matches!(res.kind, ir::ResourceKind::StorageRead)
            || (class == StorageClass::StorageBuffer
                && matches!(res.kind, ir::ResourceKind::Uniform { .. }));
        if read_only {
            self.decorate(var, Decoration::NonWritable, &[]);
        }
        if matches!(res.kind, ir::ResourceKind::StorageTexture { .. }) {
            self.decorate(var, Decoration::NonReadable, &[]);
        }
        self.resources[r] = Some((var, class));
        Ok((var, class))
    }

    /// The table's array variable holding images (or samplers) of type `t`, declared the first
    /// time: in the table's set at the binding of its kind ([`wrela_abi::table`]), aliasing every
    /// other of that kind.
    fn table_var(&mut self, t: Word, storage: bool, sampler: bool) -> Word {
        if let Some(&v) = self.tables.get(&t) {
            return v;
        }
        self.capability(Capability::RuntimeDescriptorArray);
        let array = self.type_id(TypeKey::DescriptorArray(t));
        let ptr = self.pointer(StorageClass::UniformConstant, array);
        let var = self.id();
        self.globals.extend(inst(Op::Variable, &[ptr, var, StorageClass::UniformConstant as u32]));
        let binding = if sampler {
            wrela_abi::table::SAMPLERS
        } else if storage {
            wrela_abi::table::STORAGE
        } else {
            wrela_abi::table::SAMPLED
        };
        self.name(var, &format!("table{binding}_{t}"));
        self.decorate(var, Decoration::DescriptorSet, &[wrela_abi::table::TABLE_SET]);
        self.decorate(var, Decoration::Binding, &[binding]);
        if storage {
            self.decorate(var, Decoration::NonReadable, &[]);
        }
        self.tables.insert(t, var);
        var
    }

    /// The push constants' variable, the block of indices, declared the first time: a `u32`
    /// for each binding the table holds, at its offset ([`wrela_abi::table::indexed`]).
    fn push(&mut self) -> Word {
        if let Some(v) = self.push {
            return v;
        }
        let u = self.u32_ty();
        let n = wrela_abi::table::indexed(self.p).len();
        let members = vec![u; n];
        let offsets: Vec<u32> = (0..n as u32).map(|k| 4 * k).collect();
        let block = self.wrapper("indices", &members, Some(&offsets));
        self.decorate(block, Decoration::Block, &[]);
        let ptr = self.pointer(StorageClass::PushConstant, block);
        let var = self.id();
        self.globals.extend(inst(Op::Variable, &[ptr, var, StorageClass::PushConstant as u32]));
        self.name(var, "indices");
        self.push = Some(var);
        var
    }

    // ---- the module ----

    fn module(&mut self) -> R<()> {
        let m = self.m;
        if m.types.iter().any(|(_, d)| {
            matches!(
                d,
                ir::TypeDef::Scalar(ir::Scalar::F16) | ir::TypeDef::Vector(ir::Scalar::F16, _)
            )
        }) {
            self.capability(Capability::Float16);
        }
        let entries: std::collections::HashSet<ir::FuncId> =
            m.entry_points.iter().map(|e| e.function).collect();
        // The functions that stay calls, callees first.
        let mut done = std::collections::HashSet::new();
        let mut order = Vec::new();
        fn helpers_of(
            m: &ir::Module,
            f: ir::FuncId,
            entries: &std::collections::HashSet<ir::FuncId>,
            done: &mut std::collections::HashSet<ir::FuncId>,
            order: &mut Vec<ir::FuncId>,
        ) {
            if !done.insert(f) {
                return;
            }
            for g in ir::visit::calls(&m.functions[f.index()].body) {
                helpers_of(m, g, entries, done, order);
            }
            if !entries.contains(&f) {
                order.push(f);
            }
        }
        for e in &m.entry_points {
            helpers_of(m, e.function, &entries, &mut done, &mut order);
        }
        for f in order {
            let words = FunctionWriter::write(self, f, None)?;
            self.functions.extend(words);
        }
        for e in &m.entry_points {
            let words = FunctionWriter::write(self, e.function, Some(e))?;
            self.functions.extend(words);
        }
        Ok(())
    }
}

/// The resources each function uses, directly or through its calls.
fn resource_uses(m: &ir::Module) -> Vec<BTreeSet<usize>> {
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
                if let Some(e) = s.expr() {
                    match e {
                        ir::Expr::Texture(_, t, s, _) => {
                            used.insert(t.index());
                            if let Some(s) = s {
                                used.insert(s.index());
                            }
                        }
                        ir::Expr::TextureStore(t, _) => {
                            used.insert(t.index());
                        }
                        _ => {}
                    }
                }
            });
            used
        })
        .collect();
    (0..m.functions.len())
        .map(|f| {
            let mut all = direct[f].clone();
            for g in ir::visit::reachable(m, ir::FuncId(f as u32), &[]) {
                all.extend(direct[g.index()].iter().copied());
            }
            all
        })
        .collect()
}

fn image_dim(dim: ir::TexDim) -> (Dim, bool) {
    match dim {
        ir::TexDim::D2 => (Dim::Dim2D, false),
        ir::TexDim::D2Array => (Dim::Dim2D, true),
        ir::TexDim::Cube => (Dim::DimCube, false),
        ir::TexDim::D3 => (Dim::Dim3D, false),
    }
}

/// A matrix's rows, or an array of matrices' (what a member's `MatrixStride` is for).
fn matrix_rows(types: &ir::Types, t: ir::TypeId) -> Option<u8> {
    match types.get(t) {
        ir::TypeDef::Matrix(_, r) => Some(*r),
        ir::TypeDef::Array(e, _) => matrix_rows(types, *e),
        _ => None,
    }
}

// ---- functions -------------------------------------------------------------------------------------

/// Where `break` and `continue` go in the loop a statement is in.
#[derive(Clone, Copy)]
struct LoopTargets {
    merge: Word,
    continuing: Word,
}

/// A place as a pointer: its id, its storage class, whether its type is laid out (in a buffer),
/// its IR type and whether it's an atomic.
struct Pointer {
    id: Word,
    class: StorageClass,
    laid: bool,
    ty: ir::TypeId,
}

/// An entry point's outputs: a vertex shader's varyings (each its variable) and its point size,
/// or a fragment shader's colours and depth.
enum Outputs {
    Vertex { varyings: Vec<Word>, point_size: Word },
    Fragment { colours: Vec<Word>, depth: Option<Word> },
}

struct FunctionWriter<'a, 'm> {
    w: &'a mut Writer<'m>,
    f: &'m ir::Function,
    /// Its float values' bounds (`wrela_ir::ranges`): a conversion of a bounded one isn't clamped.
    ranges: wrela_ir::ranges::Ranges<'m>,
    entry: Option<&'m ir::EntryPoint>,
    /// The function's variables, written in its first block.
    vars: Vec<Word>,
    body: Vec<Word>,
    /// Whether the block being written is still open (no terminator yet).
    open: bool,
    vals: Vec<Option<Value>>,
    uses: Vec<u32>,
    locals: Vec<Word>,
    params: Vec<Word>,
    /// Each builtin input's variable and its type.
    inputs: Vec<Option<(Word, Word)>>,
    /// A fragment shader's varyings' variables (`None` for the position, which is FragCoord) and
    /// the FragCoord variable.
    varyings_in: Vec<Option<(Word, Word)>>,
    frag_coord: Option<(Word, Word)>,
    outputs: Option<Outputs>,
    /// The global variables the entry point reaches: its interface.
    interface: BTreeSet<Word>,
    /// The integer constants among its values.
    consts: HashMap<ir::ValueId, i64>,
    /// The variable each table is stored into before a read.
    table_vars: HashMap<ir::DataId, Word>,
    /// Each texture or sampler of the device's table it reaches: a pointer to its element, made
    /// at its top.
    elements: HashMap<usize, Word>,
}

impl<'a, 'm> FunctionWriter<'a, 'm> {
    fn write(
        w: &'a mut Writer<'m>,
        id: ir::FuncId,
        entry: Option<&'m ir::EntryPoint>,
    ) -> R<Vec<Word>> {
        let f = &w.m.functions[id.index()];
        let mut uses = vec![0u32; f.values.len()];
        ir::visit::walk(&f.body, &mut |s| {
            s.for_each_value(&mut |v| uses[v.index()] += 1);
        });
        let mut fw = FunctionWriter {
            w,
            f,
            ranges: wrela_ir::ranges::Ranges::new(f),
            entry,
            vars: Vec::new(),
            body: Vec::new(),
            open: true,
            vals: vec![None; f.values.len()],
            uses,
            locals: Vec::new(),
            params: Vec::new(),
            inputs: Vec::new(),
            varyings_in: Vec::new(),
            frag_coord: None,
            outputs: None,
            interface: BTreeSet::new(),
            consts: HashMap::new(),
            table_vars: HashMap::new(),
            elements: HashMap::new(),
        };
        let fid = fw.w.function_ids[id.index()];
        let mut head = Vec::new();
        // The signature: an entry point's is `void()`, its inputs and outputs variables.
        let (ret, param_types) = match entry {
            Some(_) => (fw.w.void(), Vec::new()),
            None => {
                let ret = match f.ret {
                    Some(r) => fw.w.ty(r, false)?,
                    None => fw.w.void(),
                };
                let mut ps = Vec::new();
                for p in &f.params {
                    if p.by_ref {
                        return Err("internal: a place passed to a called GPU function".into());
                    }
                    ps.push(fw.w.ty(p.ty, false)?);
                }
                (ret, ps)
            }
        };
        let fn_ty = fw.w.type_id(TypeKey::Function(ret, param_types.clone()));
        head.extend(inst(Op::Function, &[ret, fid, 0, fn_ty]));
        fw.w.name(fid, &ir::ident(&f.name));
        for t in param_types {
            let pid = fw.w.id();
            head.extend(inst(Op::FunctionParameter, &[t, pid]));
            fw.params.push(pid);
        }
        let label = fw.w.id();
        head.extend(inst(Op::Label, &[label]));
        // Locals, zeroed as WGSL's are.
        for l in &f.locals {
            let t = fw.w.ty(l.ty, false)?;
            let var = fw.var(t);
            fw.locals.push(var);
        }
        // The resources it reaches are its interface; one of the device's table is its element
        // there, at the index the push constants give.
        for r in fw.w.uses[id.index()].clone() {
            let (var, _) = fw.w.resource(r)?;
            fw.interface.insert(var);
            let binding = fw.w.m.resources[r].binding;
            if let Some(at) = wrela_abi::table::index_offset(fw.w.p, binding) {
                let push = fw.w.push();
                fw.interface.insert(push);
                let u = fw.w.u32_ty();
                let pu = fw.w.pointer(StorageClass::PushConstant, u);
                let k = fw.w.const_u32(at / 4);
                let ptr = fw.op(Op::AccessChain, pu, &[push, k]);
                let index = fw.op(Op::Load, u, &[ptr]);
                let (t, _) = fw.w.pointee(r)?;
                let pt = fw.w.pointer(StorageClass::UniformConstant, t);
                let element = fw.op(Op::AccessChain, pt, &[var, index]);
                fw.elements.insert(r, element);
            }
        }
        if let Some(e) = entry {
            fw.entry_prologue(e)?;
        }
        // (The IR's own body, not a copy: what the compiler proves of its loops is by address.)
        let body = &f.body;
        fw.block(body, None)?;
        if fw.open {
            if let Some(ir::Stmt::Trap) = body.last() {
                fw.return_zero()?;
            } else if entry.is_some() || f.ret.is_none() {
                fw.emit(Op::Return, &[]);
                fw.open = false;
            } else {
                fw.emit(Op::Unreachable, &[]);
                fw.open = false;
            }
        }
        let mut out = head;
        out.extend_from_slice(&fw.vars);
        out.extend_from_slice(&fw.body);
        out.extend(inst(Op::FunctionEnd, &[]));
        if let Some(e) = entry {
            fw.entry_point(e, fid)?;
        }
        Ok(out)
    }

    // ---- writing ----

    fn emit(&mut self, op: Op, operands: &[Word]) {
        self.body.extend(inst(op, operands));
    }

    /// A new id, from the module's count.
    fn id(&mut self) -> Word {
        self.w.id()
    }

    /// An instruction with a result: its id.
    fn op(&mut self, op: Op, ty: Word, operands: &[Word]) -> Word {
        let id = self.id();
        let mut all = vec![ty, id];
        all.extend_from_slice(operands);
        self.emit(op, &all);
        id
    }

    fn ext(&mut self, gl: GLOp, ty: Word, args: &[Word]) -> Word {
        let glsl = self.w.glsl;
        let mut operands = vec![glsl, gl as u32];
        operands.extend_from_slice(args);
        self.op(Op::ExtInst, ty, &operands)
    }

    /// A variable of the function, of type `t`, zeroed.
    fn var(&mut self, t: Word) -> Word {
        let ptr = self.w.pointer(StorageClass::Function, t);
        let null = self.w.null(t);
        let id = self.id();
        self.vars.extend(inst(Op::Variable, &[ptr, id, StorageClass::Function as u32, null]));
        id
    }

    /// Starts block `label`, after the last was ended.
    fn label(&mut self, label: Word) {
        self.emit(Op::Label, &[label]);
        self.open = true;
    }

    fn branch(&mut self, target: Word) {
        if self.open {
            self.emit(Op::Branch, &[target]);
            self.open = false;
        }
    }

    // ---- statements ----

    fn block(&mut self, b: &[ir::Stmt], loop_targets: Option<LoopTargets>) -> R<()> {
        for s in b {
            if !self.open {
                // What follows a branch away is never reached.
                break;
            }
            self.stmt(s, loop_targets)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &ir::Stmt, lt: Option<LoopTargets>) -> R<()> {
        match s {
            ir::Stmt::Let(v, e) => {
                if self.uses[v.index()] == 0 && pure(e) {
                    return Ok(());
                }
                let value = self.value(*v, e)?;
                self.vals[v.index()] = Some(value);
            }
            ir::Stmt::Eval(e) => match e {
                ir::Expr::Call(g, args) => {
                    self.call(*g, args)?;
                }
                ir::Expr::Host(..) => return Err("GPU code can't record GPU work".into()),
                ir::Expr::Mem(..) => return Err("GPU code can't use raw memory".into()),
                ir::Expr::Barrier => {
                    let scope = self.w.const_u32(spirv::Scope::Workgroup as u32);
                    let semantics = self.w.const_u32(0x108);
                    self.emit(Op::ControlBarrier, &[scope, scope, semantics]);
                }
                ir::Expr::Discard => {
                    self.emit(Op::Kill, &[]);
                    self.open = false;
                }
                ir::Expr::Atomic(op, p, args) => {
                    self.atomic(*op, p, args, None)?;
                }
                ir::Expr::TextureStore(tex, args) => self.texture_store(*tex, args)?,
                _ => {}
            },
            ir::Stmt::Store(p, v) => {
                let value = self.val(*v)?;
                self.store(p, value)?;
            }
            ir::Stmt::If { cond, then, else_ } => {
                let c = self.val(*cond)?;
                let merge = self.id();
                let accept = self.id();
                let reject = if else_.is_empty() { merge } else { self.id() };
                self.emit(Op::SelectionMerge, &[merge, 0]);
                self.emit(Op::BranchConditional, &[c, accept, reject]);
                self.open = false;
                self.label(accept);
                self.block(then, lt)?;
                self.branch(merge);
                if !else_.is_empty() {
                    self.label(reject);
                    self.block(else_, lt)?;
                    self.branch(merge);
                }
                self.label(merge);
            }
            ir::Stmt::Loop { body, continuing } => {
                let header = self.id();
                let merge = self.id();
                let continue_id = self.id();
                let body_id = self.id();
                self.branch(header);
                self.label(header);
                self.emit(Op::LoopMerge, &[merge, continue_id, 0]);
                if self.w.checks.loops && !self.w.proofs.ends(body) {
                    let check = self.id();
                    self.branch(check);
                    self.label(check);
                    self.loop_bound(merge)?;
                }
                self.branch(body_id);
                self.label(body_id);
                let targets = LoopTargets { merge, continuing: continue_id };
                self.block(body, Some(targets))?;
                self.branch(continue_id);
                self.label(continue_id);
                self.block(continuing, None)?;
                self.branch(header);
                self.label(merge);
            }
            ir::Stmt::Break => {
                let t = lt.ok_or("internal: a break outside a loop")?;
                self.branch(t.merge);
            }
            ir::Stmt::Continue => {
                let t = lt.ok_or("internal: a continue outside a loop")?;
                self.branch(t.continuing);
            }
            ir::Stmt::Return(v) => {
                let value = match v {
                    Some(v) => Some(self.val(*v)?),
                    None => None,
                };
                self.ret(value)?;
            }
            ir::Stmt::At(_) | ir::Stmt::Trap => {}
        }
        Ok(())
    }

    /// The loop's bound, naga's: a 64-bit count down from the most, in two `u32`s, which leaves
    /// the loop at zero; the GPU's compiler then can't assume the loop ends.
    fn loop_bound(&mut self, merge: Word) -> R<()> {
        let u = self.w.u32_ty();
        let u2 = self.w.vec_ty(u, 2);
        let b = self.w.bool_ty();
        let b2 = self.w.vec_ty(b, 2);
        let zero = self.w.const_u32(0);
        let one = self.w.const_u32(1);
        let max = self.w.const_u32(u32::MAX);
        let zero2 = self.w.constant(ConstKey::Composite(u2, vec![zero, zero]));
        let max2 = self.w.constant(ConstKey::Composite(u2, vec![max, max]));
        let ptr = self.w.pointer(StorageClass::Function, u2);
        let counter = self.id();
        self.vars.extend(inst(Op::Variable, &[ptr, counter, StorageClass::Function as u32, max2]));
        let load = self.op(Op::Load, u2, &[counter]);
        let eq = self.op(Op::IEqual, b2, &[zero2, load]);
        let all = self.op(Op::All, b, &[eq]);
        let inc = self.id();
        self.emit(Op::SelectionMerge, &[inc, 0]);
        self.emit(Op::BranchConditional, &[all, merge, inc]);
        self.open = false;
        self.label(inc);
        let low = self.op(Op::CompositeExtract, u, &[load, 1]);
        let low_zero = self.op(Op::IEqual, b, &[low, zero]);
        let carry = self.op(Op::Select, u, &[low_zero, one, zero]);
        let dec = self.op(Op::CompositeConstruct, u2, &[carry, one]);
        let next = self.op(Op::ISub, u2, &[load, dec]);
        self.emit(Op::Store, &[counter, next]);
        Ok(())
    }

    fn call(&mut self, g: ir::FuncId, args: &[ir::Arg]) -> R<Word> {
        let mut ids = Vec::new();
        for a in args {
            match a {
                ir::Arg::Value(v) => ids.push(self.val(*v)?),
                ir::Arg::Place(_) => {
                    return Err("internal: a place passed to a called GPU function".into());
                }
            }
        }
        let callee = &self.w.m.functions[g.index()];
        let ret = match callee.ret {
            Some(r) => self.w.ty(r, false)?,
            None => self.w.void(),
        };
        let fid = self.w.function_ids[g.index()];
        let mut operands = vec![fid];
        operands.extend(ids);
        Ok(self.op(Op::FunctionCall, ret, &operands))
    }

    /// `return`: a helper's value, or an entry point's outputs written.
    fn ret(&mut self, value: Option<Word>) -> R<()> {
        match (&self.outputs, value) {
            (None, Some(v)) => {
                self.emit(Op::ReturnValue, &[v]);
            }
            (None, None) => self.emit(Op::Return, &[]),
            (Some(_), v) => {
                if let Some(v) = v {
                    self.write_outputs(v)?;
                }
                self.emit(Op::Return, &[]);
            }
        }
        self.open = false;
        Ok(())
    }

    /// A return of a zero of the function's result (or its outputs zeroed).
    fn return_zero(&mut self) -> R<()> {
        match (self.entry, self.f.ret) {
            (Some(_), Some(r)) => {
                let t = self.w.ty(r, false)?;
                let zero = self.w.null(t);
                self.ret(Some(zero))
            }
            (None, Some(r)) => {
                let t = self.w.ty(r, false)?;
                let zero = self.w.null(t);
                self.ret(Some(zero))
            }
            (_, None) => self.ret(None),
        }
    }

    fn val(&self, v: ir::ValueId) -> R<Word> {
        self.vals
            .get(v.index())
            .copied()
            .flatten()
            .map(|v| v.id)
            .ok_or_else(|| format!("internal: v{} has no SPIR-V id", v.0))
    }

    fn is_const(&self, v: ir::ValueId) -> bool {
        self.vals.get(v.index()).copied().flatten().is_some_and(|v| v.constant)
    }

    fn ty_of(&self, v: ir::ValueId) -> ir::TypeId {
        self.f.value_ty(v)
    }

    fn scalar_of(&self, v: ir::ValueId) -> Option<ir::Scalar> {
        self.w.m.types.element_scalar(self.ty_of(v))
    }

    /// How many components `v`'s type has (1 for a scalar).
    fn width_of(&self, v: ir::ValueId) -> u32 {
        match self.w.m.types.get(self.ty_of(v)) {
            ir::TypeDef::Vector(_, n) => u32::from(*n),
            _ => 1,
        }
    }
}

/// Whether `e` only computes, with nothing to do if nothing reads it.
fn pure(e: &ir::Expr) -> bool {
    e.only_computes()
        && !matches!(
            e,
            ir::Expr::ExtractDyn(..)
                | ir::Expr::Builtin(ir::Builtin::Dpdx | ir::Builtin::Dpdy | ir::Builtin::Fwidth, _)
        )
}

// ---- entry points ------------------------------------------------------------------------------------

impl<'m> FunctionWriter<'_, 'm> {
    /// An input or output variable of type `t`.
    fn io_var(&mut self, class: StorageClass, t: Word, name: &str) -> Word {
        let ptr = self.w.pointer(class, t);
        let id = self.w.id();
        self.w.globals.extend(inst(Op::Variable, &[ptr, id, class as u32]));
        self.w.name(id, name);
        self.interface.insert(id);
        id
    }

    fn builtin_var(&mut self, class: StorageClass, t: Word, b: BuiltIn, name: &str) -> Word {
        let id = self.io_var(class, t, name);
        self.w.decorate(id, Decoration::BuiltIn, &[b as u32]);
        id
    }

    /// An entry point's inputs and outputs, and a kernel's workgroup memory zeroed.
    fn entry_prologue(&mut self, e: &'m ir::EntryPoint) -> R<()> {
        let in_varyings = |b: &ir::BuiltinInput| {
            *b == ir::BuiltinInput::Position
                && matches!(e.stage, ir::Stage::Fragment { varyings: Some(_), .. })
        };
        let u = self.w.u32_ty();
        let u3 = self.w.vec_ty(u, 3);
        let f = self.w.f32_ty();
        let f4 = self.w.vec_ty(f, 4);
        let mut local_index = None;
        for b in &e.inputs {
            use ir::BuiltinInput as B;
            if in_varyings(b) {
                self.inputs.push(None);
                continue;
            }
            let (t, builtin, name) = match b {
                B::GlobalInvocationId => (u3, BuiltIn::GlobalInvocationId, "global_id"),
                B::LocalInvocationId => (u3, BuiltIn::LocalInvocationId, "local_id"),
                B::WorkgroupId => (u3, BuiltIn::WorkgroupId, "workgroup_id"),
                B::NumWorkgroups => (u3, BuiltIn::NumWorkgroups, "num_workgroups"),
                B::VertexIndex => (u, BuiltIn::VertexIndex, "vertex_index"),
                B::InstanceIndex => (u, BuiltIn::InstanceIndex, "instance_index"),
                B::Position => (f4, BuiltIn::FragCoord, "frag_coord"),
                B::SubgroupInvocationId => {
                    self.w.capability(Capability::GroupNonUniform);
                    (u, BuiltIn::SubgroupLocalInvocationId, "subgroup_invocation_id")
                }
                B::SubgroupSize => {
                    self.w.capability(Capability::GroupNonUniform);
                    (u, BuiltIn::SubgroupSize, "subgroup_size")
                }
            };
            let var = self.builtin_var(StorageClass::Input, t, builtin, name);
            if *b == B::Position {
                self.frag_coord = Some((var, f4));
            }
            self.inputs.push(Some((var, t)));
        }
        match &e.stage {
            ir::Stage::Compute { .. } => {
                // Workgroup memory zeroed by the first invocation, before a barrier.
                let workgroup: Vec<usize> = self.w.uses[e.function.index()]
                    .iter()
                    .copied()
                    .filter(|&r| self.w.m.resources[r].kind == ir::ResourceKind::Workgroup)
                    .collect();
                if !workgroup.is_empty() {
                    let var = self.builtin_var(
                        StorageClass::Input,
                        u,
                        BuiltIn::LocalInvocationIndex,
                        "local_index",
                    );
                    local_index = Some(var);
                    let index = self.op(Op::Load, u, &[var]);
                    let zero = self.w.const_u32(0);
                    let b = self.w.bool_ty();
                    let first = self.op(Op::IEqual, b, &[index, zero]);
                    let (accept, merge) = (self.id(), self.id());
                    self.emit(Op::SelectionMerge, &[merge, 0]);
                    self.emit(Op::BranchConditional, &[first, accept, merge]);
                    self.open = false;
                    self.label(accept);
                    for r in workgroup {
                        let (var, _) = self.w.resource(r)?;
                        let t = self.w.ty(self.w.m.resources[r].ty, false)?;
                        let null = self.w.null(t);
                        self.emit(Op::Store, &[var, null]);
                    }
                    self.branch(merge);
                    self.label(merge);
                    let scope = self.w.const_u32(spirv::Scope::Workgroup as u32);
                    let semantics = self.w.const_u32(0x108);
                    self.emit(Op::ControlBarrier, &[scope, scope, semantics]);
                }
            }
            ir::Stage::Vertex { varyings } => {
                let rt = self.f.ret.ok_or("internal: a vertex shader with no output")?;
                let mut vars = Vec::new();
                let mut location = 0;
                for (k, v) in varyings.iter().enumerate() {
                    let pt = part_ty(&self.w.m.types, rt, &v.path)?;
                    let t = self.w.ty(pt, false)?;
                    if v.position {
                        vars.push(self.builtin_var(
                            StorageClass::Output,
                            t,
                            BuiltIn::Position,
                            "position",
                        ));
                    } else {
                        let var = self.io_var(StorageClass::Output, t, &format!("v{k}"));
                        self.w.decorate(var, Decoration::Location, &[location]);
                        if v.flat {
                            self.w.decorate(var, Decoration::Flat, &[]);
                        }
                        location += 1;
                        vars.push(var);
                    }
                }
                let point_size =
                    self.builtin_var(StorageClass::Output, f, BuiltIn::PointSize, "point_size");
                self.outputs = Some(Outputs::Vertex { varyings: vars, point_size });
            }
            ir::Stage::Fragment { varyings, outputs, depth, .. } => {
                if let Some((vt, map)) = varyings {
                    let mut location = 0;
                    for (k, v) in map.iter().enumerate() {
                        if v.position {
                            if self.frag_coord.is_none() {
                                let var = self.builtin_var(
                                    StorageClass::Input,
                                    f4,
                                    BuiltIn::FragCoord,
                                    "frag_coord",
                                );
                                self.frag_coord = Some((var, f4));
                            }
                            self.varyings_in.push(None);
                            continue;
                        }
                        let pt = part_ty(&self.w.m.types, *vt, &v.path)?;
                        let t = self.w.ty(pt, false)?;
                        let var = self.io_var(StorageClass::Input, t, &format!("v{k}"));
                        self.w.decorate(var, Decoration::Location, &[location]);
                        if v.flat {
                            self.w.decorate(var, Decoration::Flat, &[]);
                        }
                        location += 1;
                        self.varyings_in.push(Some((var, t)));
                    }
                }
                let mut colours = Vec::new();
                for (location, c) in outputs.iter().enumerate() {
                    let s = self.w.scalar_ty(c.scalar)?;
                    let t = self.w.vec_ty(s, 4);
                    let var = self.io_var(StorageClass::Output, t, &format!("color{location}"));
                    self.w.decorate(var, Decoration::Location, &[location as u32]);
                    colours.push(var);
                }
                let depth = depth.as_ref().map(|_| {
                    self.builtin_var(StorageClass::Output, f, BuiltIn::FragDepth, "frag_depth")
                });
                self.outputs = Some(Outputs::Fragment { colours, depth });
            }
        }
        let _ = local_index;
        Ok(())
    }

    /// Writes `value`, the entry point's result, to its outputs.
    fn write_outputs(&mut self, value: Word) -> R<()> {
        let e = self.entry.ok_or("internal: outputs outside an entry point")?;
        let rt = self.f.ret.ok_or("internal: an entry point's output with no type")?;
        match (&e.stage, self.outputs.as_ref()) {
            (
                ir::Stage::Vertex { varyings },
                Some(Outputs::Vertex { varyings: vars, point_size }),
            ) => {
                let (vars, point_size) = (vars.clone(), *point_size);
                for (v, var) in varyings.iter().zip(vars) {
                    let part = self.part(value, rt, &v.path)?;
                    self.emit(Op::Store, &[var, part]);
                }
                let one = self.w.const_f32(1.0);
                self.emit(Op::Store, &[point_size, one]);
            }
            (
                ir::Stage::Fragment { outputs, depth, .. },
                Some(Outputs::Fragment { colours, depth: dvar }),
            ) => {
                let (colours, dvar) = (colours.clone(), *dvar);
                for (c, var) in outputs.iter().zip(colours) {
                    let part = self.part(value, rt, &c.path)?;
                    let widened = self.widened(part, c)?;
                    self.emit(Op::Store, &[var, widened]);
                }
                if let (Some(path), Some(var)) = (depth, dvar) {
                    let part = self.part(value, rt, path)?;
                    self.emit(Op::Store, &[var, part]);
                }
            }
            _ => return Err("internal: an entry point's outputs".into()),
        }
        Ok(())
    }

    /// The part of `value` (of IR type `t`) at `path`: field indexes, then a matrix column's.
    fn part(&mut self, value: Word, t: ir::TypeId, path: &[u32]) -> R<Word> {
        let mut id = value;
        let mut t = t;
        for &k in path {
            let member = member(&self.w.m.types, t, k);
            let next = self.w.m.types.part(t, k).ok_or("internal: a varying's path")?;
            let nt = self.w.ty(next, false)?;
            id = self.op(Op::CompositeExtract, nt, &[id, member]);
            t = next;
        }
        Ok(id)
    }

    /// A colour of `c.components` of `c.scalar` as four of it: missing colour channels 0, and a
    /// float's missing alpha 1.
    fn widened(&mut self, part: Word, c: &ir::FragmentColour) -> R<Word> {
        if c.components == 4 {
            return Ok(part);
        }
        let s = self.w.scalar_ty(c.scalar)?;
        let four = self.w.vec_ty(s, 4);
        let mut parts = Vec::new();
        if c.components == 1 {
            parts.push(part);
        } else {
            for k in 0..u32::from(c.components) {
                parts.push(self.op(Op::CompositeExtract, s, &[part, k]));
            }
        }
        for k in c.components..4 {
            parts.push(match (c.scalar, k) {
                (ir::Scalar::U32, _) => self.w.const_u32(0),
                (_, 3) => self.w.const_f32(1.0),
                _ => self.w.const_f32(0.0),
            });
        }
        Ok(self.op(Op::CompositeConstruct, four, &parts))
    }

    /// The part at `path` of a fragment shader's input, of IR type `t`, rebuilt from its input
    /// variables.
    fn rebuild(&mut self, map: &[ir::Varying], t: ir::TypeId, path: &mut Vec<u32>) -> R<Word> {
        if let Some(k) = map.iter().position(|v| v.path == *path) {
            if map[k].position {
                let (var, ty) = self.frag_coord.ok_or("internal: no frag coord")?;
                return Ok(self.op(Op::Load, ty, &[var]));
            }
            let (var, ty) = self.varyings_in[k].ok_or("internal: a varying's variable")?;
            return Ok(self.op(Op::Load, ty, &[var]));
        }
        let n = match self.w.m.types.get(t) {
            ir::TypeDef::Struct { fields, .. } => fields.len() as u32,
            ir::TypeDef::Matrix(c, _) => u32::from(*c),
            _ => return Err("internal: a varying's part that no member holds".into()),
        };
        let mut parts = Vec::with_capacity(n as usize);
        for k in 0..n {
            let pt = part_ty(&self.w.m.types, t, &[k])?;
            path.push(k);
            parts.push(self.rebuild(map, pt, path)?);
            path.pop();
        }
        let ty = self.w.ty(t, false)?;
        Ok(self.op(Op::CompositeConstruct, ty, &parts))
    }

    /// The entry point's declaration and its execution modes.
    fn entry_point(&mut self, e: &ir::EntryPoint, fid: Word) -> R<()> {
        let name = ir::ident(&e.name);
        self.w.entry_names.push(name.clone());
        let model = match e.stage {
            ir::Stage::Compute { .. } => ExecutionModel::GLCompute,
            ir::Stage::Vertex { .. } => ExecutionModel::Vertex,
            ir::Stage::Fragment { .. } => ExecutionModel::Fragment,
        };
        let mut operands = vec![model as u32, fid];
        operands.extend(string(&name));
        operands.extend(self.interface.iter().copied());
        self.w.entry_points.extend(inst(Op::EntryPoint, &operands));
        match &e.stage {
            ir::Stage::Compute { workgroup_size: [x, y, z] } => {
                self.w.execution_modes.extend(inst(
                    Op::ExecutionMode,
                    &[fid, ExecutionMode::LocalSize as u32, *x, *y, *z],
                ));
            }
            ir::Stage::Vertex { .. } => {}
            ir::Stage::Fragment { depth, .. } => {
                self.w
                    .execution_modes
                    .extend(inst(Op::ExecutionMode, &[fid, ExecutionMode::OriginUpperLeft as u32]));
                if depth.is_some() {
                    self.w.execution_modes.extend(inst(
                        Op::ExecutionMode,
                        &[fid, ExecutionMode::DepthReplacing as u32],
                    ));
                    if let Some(p) = self.w.promise {
                        let mode = match p {
                            ir::DepthPromise::Greater => ExecutionMode::DepthGreater,
                            ir::DepthPromise::Less => ExecutionMode::DepthLess,
                        };
                        self.w.execution_modes.extend(inst(Op::ExecutionMode, &[fid, mode as u32]));
                    }
                }
            }
        }
        Ok(())
    }
}

/// The struct member that holds field `k` of a value of type `t`: for an enum, its tag, or the
/// payload of variant `k - 1` among the variants that have one.
fn member(types: &ir::Types, t: ir::TypeId, k: u32) -> u32 {
    match types.get(t) {
        ir::TypeDef::Enum { variants, .. } if k > 0 => {
            1 + variants[..k as usize - 1].iter().filter(|(_, p)| p.is_some()).count() as u32
        }
        _ => k,
    }
}

/// The type of the part of a value of type `t` at `path`: field indexes, then a matrix column's.
fn part_ty(types: &ir::Types, t: ir::TypeId, path: &[u32]) -> R<ir::TypeId> {
    path.iter()
        .try_fold(t, |t, &k| types.part(t, k))
        .ok_or_else(|| "internal: a varying's path".to_string())
}

/// How many elements an index into a value of type `t` can reach, when the type says.
fn static_len(types: &ir::Types, t: ir::TypeId) -> Option<u32> {
    match types.get(t) {
        ir::TypeDef::Array(_, n) => Some(*n),
        ir::TypeDef::Vector(_, n) | ir::TypeDef::Matrix(n, _) => Some(u32::from(*n)),
        _ => None,
    }
}

// ---- places ------------------------------------------------------------------------------------------

impl FunctionWriter<'_, '_> {
    /// An index into something of `len` elements, or into a sized buffer's array (`runtime`: its
    /// variable): clamped into range by `UMin` unless it's a constant known to be inside, and a
    /// `u32` either way. A buffer's index isn't clamped: the device's robust buffer access, which
    /// the native host requires, keeps every read and write inside the buffer (a read gives a
    /// value from inside it, or 0; a write lands inside it, or nowhere: WebGPU's robust access,
    /// language.md §11.3). AMD's hardware checks a buffer load against its descriptor's size for
    /// nothing, so a clamp a load was work twice over: the stack's eye on the Radeon, 8.32 ->
    /// 7.46 ms (backend.md, item 19). A debug build still checks each index (`wrela_ir::bounds`).
    fn index(&mut self, i: ir::ValueId, len: Option<u32>, runtime: Option<Word>) -> R<Word> {
        let u = self.w.u32_ty();
        let mut id = self.val(i)?;
        if self.scalar_of(i) == Some(ir::Scalar::I32) {
            id = self.op(Op::Bitcast, u, &[id]);
        }
        if !self.w.checks.bounds {
            return Ok(id);
        }
        if let (Some(n), Some(k)) = (len, self.consts.get(&i).copied())
            && (0..i64::from(n)).contains(&k)
        {
            return Ok(self.w.const_u32(k as u32));
        }
        if let Some(n) = len
            && self.w.proofs.in_range(self.f, i, n)
        {
            return Ok(id);
        }
        let max = match (len, runtime) {
            (Some(n), _) => self.w.const_u32(n - 1),
            (None, Some(_)) => return Ok(id),
            (None, None) => return Err("internal: an index with no length".into()),
        };
        Ok(self.ext(GLOp::UMin, u, &[id, max]))
    }

    /// A place as a pointer. A place rooted at a parameter (a value) has none: see [`Self::load`].
    fn pointer(&mut self, p: &ir::Place) -> R<Pointer> {
        let m = self.w.m;
        let (base, class, laid, mut t, sized) = match &p.root {
            ir::PlaceRoot::Local(l) => (
                self.locals[l.index()],
                StorageClass::Function,
                false,
                Some(self.f.locals[l.index()].ty),
                None,
            ),
            ir::PlaceRoot::Resource(r) => {
                let (var, class) = self.w.resource(r.index())?;
                let res = &m.resources[r.index()];
                let laid = matches!(class, StorageClass::Uniform | StorageClass::StorageBuffer);
                if matches!(
                    res.kind,
                    ir::ResourceKind::StorageRead | ir::ResourceKind::StorageReadWrite
                ) {
                    (var, class, true, None, Some((var, res.ty)))
                } else {
                    (var, class, laid, Some(res.ty), None)
                }
            }
            ir::PlaceRoot::Data(d) => {
                // The table stored into a variable of the function just before each read, as
                // Mesa (RADV) reads one fastest (`crate::Tables::Local`).
                let data = &m.data[d.index()];
                let ty = self.w.ty(data.ty, false)?;
                let value = self.w.const_value(data.ty, &data.value)?;
                let var = match self.table_vars.get(d) {
                    Some(&v) => v,
                    None => {
                        let v = self.var(ty);
                        self.table_vars.insert(*d, v);
                        v
                    }
                };
                self.emit(Op::Store, &[var, value]);
                (var, StorageClass::Function, false, Some(data.ty), None)
            }
            ir::PlaceRoot::Param(_) => return Err("internal: a pointer to a parameter".into()),
            ir::PlaceRoot::Ptr(_) => {
                return Err(
                    "projections returned from functions aren't supported on the GPU".into()
                );
            }
        };
        let mut indices = Vec::new();
        let mut sized = sized;
        for proj in &p.path {
            match (proj, t) {
                (ir::Proj::Index(i), None) => {
                    let (var, elem) = sized.take().ok_or("internal: a place's type")?;
                    let zero = self.w.const_u32(0);
                    indices.push(zero);
                    indices.push(self.index(*i, None, Some(var))?);
                    t = Some(elem);
                }
                (ir::Proj::Field(k), Some(pt)) => {
                    indices.push(self.w.const_u32(member(&m.types, pt, *k)));
                    t = m.proj_ty(pt, proj);
                }
                (ir::Proj::Comp(c), Some(pt)) => {
                    indices.push(self.w.const_u32(u32::from(*c)));
                    t = m.proj_ty(pt, proj);
                }
                (ir::Proj::Index(i), Some(pt)) => {
                    let len = static_len(&m.types, pt);
                    indices.push(self.index(*i, len, None)?);
                    t = m.proj_ty(pt, proj);
                }
                _ => return Err("internal: a place's type".into()),
            }
        }
        let t = t.ok_or("internal: a place's type")?;
        if indices.is_empty() {
            return Ok(Pointer { id: base, class, laid, ty: t });
        }
        let pointee = self.w.ty(t, laid)?;
        let ptr_ty = self.w.pointer(class, pointee);
        let mut operands = vec![base];
        operands.extend(indices);
        let id = self.op(Op::AccessChain, ptr_ty, &operands);
        Ok(Pointer { id, class, laid, ty: t })
    }

    fn is_aggregate(&self, t: ir::TypeId) -> bool {
        matches!(
            self.w.m.types.get(t),
            ir::TypeDef::Array(..) | ir::TypeDef::Struct { .. } | ir::TypeDef::Enum { .. }
        )
    }

    /// The scope and memory semantics of an atomic in `class`: relaxed, with no storage class
    /// named (Vulkan asks a non-relaxed order of semantics that name one,
    /// VUID-StandaloneSpirv-MemorySemantics-10871; naga named one, relaxed, which orders nothing).
    fn atomic_scope(&mut self, class: StorageClass) -> (Word, Word) {
        let scope = match class {
            StorageClass::Workgroup => spirv::Scope::Workgroup as u32,
            _ => spirv::Scope::Device as u32,
        };
        (self.w.const_u32(scope), self.w.const_u32(0))
    }

    /// A place's value: through its pointer, or (a parameter's) taken apart.
    fn load(&mut self, p: &ir::Place) -> R<Word> {
        if let ir::PlaceRoot::Param(i) = p.root {
            return self.param_part(i, &p.path);
        }
        let ptr = self.pointer(p)?;
        let plain = self.w.ty(ptr.ty, false)?;
        if matches!(self.w.m.types.get(ptr.ty), ir::TypeDef::Atomic(_)) {
            let (scope, semantics) = self.atomic_scope(ptr.class);
            return Ok(self.op(Op::AtomicLoad, plain, &[ptr.id, scope, semantics]));
        }
        if ptr.laid && self.is_aggregate(ptr.ty) {
            let laid = self.w.ty(ptr.ty, true)?;
            let value = self.op(Op::Load, laid, &[ptr.id]);
            return Ok(self.op(Op::CopyLogical, plain, &[value]));
        }
        Ok(self.op(Op::Load, plain, &[ptr.id]))
    }

    fn store(&mut self, p: &ir::Place, value: Word) -> R<()> {
        let ptr = self.pointer(p)?;
        if matches!(self.w.m.types.get(ptr.ty), ir::TypeDef::Atomic(_)) {
            let (scope, semantics) = self.atomic_scope(ptr.class);
            self.emit(Op::AtomicStore, &[ptr.id, scope, semantics, value]);
            return Ok(());
        }
        let value = if ptr.laid && self.is_aggregate(ptr.ty) {
            let laid = self.w.ty(ptr.ty, true)?;
            self.op(Op::CopyLogical, laid, &[value])
        } else {
            value
        };
        self.emit(Op::Store, &[ptr.id, value]);
        Ok(())
    }

    /// The part of parameter `i` (a value) at `path`: each part extracted, an element at a
    /// worked-out index read through a variable it's stored in.
    fn param_part(&mut self, i: u32, path: &[ir::Proj]) -> R<Word> {
        let mut id = self.params.get(i as usize).copied().ok_or("internal: a parameter")?;
        let mut t = self.f.params[i as usize].ty;
        for proj in path {
            let m = self.w.m;
            let next = m.proj_ty(t, proj).ok_or("internal: a parameter's part")?;
            let nt = self.w.ty(next, false)?;
            id = match proj {
                ir::Proj::Field(k) => {
                    self.op(Op::CompositeExtract, nt, &[id, member(&m.types, t, *k)])
                }
                ir::Proj::Comp(c) => self.op(Op::CompositeExtract, nt, &[id, u32::from(*c)]),
                ir::Proj::Index(v) => self.dynamic_part(id, t, *v)?,
            };
            t = next;
        }
        Ok(id)
    }

    /// The element at index `i` of `base`, a value of type `t`: a vector's by
    /// `OpVectorExtractDynamic`; an array's or a matrix's through a variable it's stored in.
    fn dynamic_part(&mut self, base: Word, t: ir::TypeId, i: ir::ValueId) -> R<Word> {
        let m = self.w.m;
        let len = static_len(&m.types, t);
        let index = self.index(i, len, None)?;
        let elem = m.proj_ty(t, &ir::Proj::Index(i)).ok_or("internal: an element's type")?;
        let et = self.w.ty(elem, false)?;
        if matches!(m.types.get(t), ir::TypeDef::Vector(..)) {
            return Ok(self.op(Op::VectorExtractDynamic, et, &[base, index]));
        }
        let bt = self.w.ty(t, false)?;
        let spill = self.var(bt);
        self.emit(Op::Store, &[spill, base]);
        let ptr = self.w.pointer(StorageClass::Function, et);
        let at = self.op(Op::AccessChain, ptr, &[spill, index]);
        Ok(self.op(Op::Load, et, &[at]))
    }
}

// ---- expressions -------------------------------------------------------------------------------------

impl FunctionWriter<'_, '_> {
    fn value(&mut self, v: ir::ValueId, e: &ir::Expr) -> R<Value> {
        let t = self.f.value_ty(v);
        let m = self.w.m;
        let computed = |id| Value { id, constant: false };
        Ok(match e {
            ir::Expr::Const(c) => {
                match c {
                    ir::Const::I32(x) => {
                        self.consts.insert(v, i64::from(*x));
                    }
                    ir::Const::U32(x) => {
                        self.consts.insert(v, i64::from(*x));
                    }
                    _ => {}
                }
                Value { id: self.w.scalar_const(c)?, constant: true }
            }
            ir::Expr::Zero(zt) => {
                let ty = self.w.ty(*zt, false)?;
                Value { id: self.w.null(ty), constant: true }
            }
            ir::Expr::Param(i) => {
                if let (
                    Some(ir::EntryPoint {
                        stage: ir::Stage::Fragment { varyings: Some((vt, map)), .. },
                        ..
                    }),
                    true,
                ) = (self.entry, true)
                {
                    let mut path = Vec::new();
                    computed(self.rebuild(map, *vt, &mut path)?)
                } else {
                    computed(*self.params.get(*i as usize).ok_or("internal: a parameter")?)
                }
            }
            ir::Expr::EntryInput(i) => {
                let (var, vt) = match self.inputs.get(*i as usize).copied() {
                    Some(Some(x)) => x,
                    Some(None) => self.frag_coord.ok_or("internal: no frag coord")?,
                    None => return Err("internal: a missing entry input".into()),
                };
                let arg = self.op(Op::Load, vt, &[var]);
                let ty = self.w.ty(t, false)?;
                if ty == vt {
                    return Ok(computed(arg));
                }
                match m.types.get(t) {
                    ir::TypeDef::Struct { fields, .. } if fields.len() == 3 => {
                        let u = self.w.u32_ty();
                        let parts: Vec<Word> =
                            (0..3).map(|k| self.op(Op::CompositeExtract, u, &[arg, k])).collect();
                        computed(self.op(Op::CompositeConstruct, ty, &parts))
                    }
                    _ => computed(self.op(Op::CompositeConstruct, ty, &[arg])),
                }
            }
            ir::Expr::Load(p) => computed(self.load(p)?),
            ir::Expr::ArrayLength(p) => {
                let var = match &p.root {
                    ir::PlaceRoot::Resource(r) => self.w.resource(r.index())?.0,
                    _ => return Err("internal: the length of something not a sized buffer".into()),
                };
                let u = self.w.u32_ty();
                computed(self.op(Op::ArrayLength, u, &[var, 0]))
            }
            ir::Expr::Texture(op, tex, sampler, args) => {
                computed(self.texture(*op, *tex, *sampler, args, t)?)
            }
            ir::Expr::Atomic(op, p, args) => {
                computed(self.atomic(*op, p, args, Some(t))?.ok_or("internal: an atomic's value")?)
            }
            ir::Expr::Barrier => return Err("internal: a barrier's value".into()),
            ir::Expr::Subgroup(op, args) => computed(self.subgroup(*op, args, t)?),
            ir::Expr::TextureStore(..) => return Err("internal: a texel write's value".into()),
            ir::Expr::Discard => return Err("internal: a discard's value".into()),
            ir::Expr::Call(g, args) => {
                if m.functions[g.index()].ret.is_none() {
                    return Err(
                        "internal: a call's value from a function that returns nothing".into()
                    );
                }
                computed(self.call(*g, args)?)
            }
            ir::Expr::Construct(ty, parts) => {
                let ids = parts.iter().map(|p| self.val(*p)).collect::<R<Vec<_>>>()?;
                let sid = self.w.ty(*ty, false)?;
                // A vector made of one whole vector of its type is that vector.
                if let ([one], ir::TypeDef::Vector(..)) = (&parts[..], m.types.get(*ty))
                    && self.ty_of(*one) == *ty
                {
                    return Ok(self.vals[one.index()].unwrap_or(computed(ids[0])));
                }
                if parts.iter().all(|p| self.is_const(*p)) {
                    Value { id: self.w.constant(ConstKey::Composite(sid, ids)), constant: true }
                } else {
                    computed(self.op(Op::CompositeConstruct, sid, &ids))
                }
            }
            ir::Expr::Extract(x, i) => {
                let xt = self.ty_of(*x);
                let base = self.val(*x)?;
                let ty = self.w.ty(t, false)?;
                let index = member(&m.types, xt, *i);
                computed(self.op(Op::CompositeExtract, ty, &[base, index]))
            }
            ir::Expr::Variant(et, k, payload) => {
                let ir::TypeDef::Enum { variants, .. } = m.types.get(*et) else {
                    return Err("internal: a variant of a type that isn't an enum".into());
                };
                let ty = self.w.ty(*et, false)?;
                let mut ids = vec![self.w.const_u32(m.types.tag(*et, *k))];
                for (vk, (_, p)) in variants.iter().enumerate() {
                    let Some(pt) = p else { continue };
                    ids.push(match payload {
                        Some(x) if vk == *k as usize => self.val(*x)?,
                        _ => {
                            let pty = self.w.ty(*pt, false)?;
                            self.w.null(pty)
                        }
                    });
                }
                computed(self.op(Op::CompositeConstruct, ty, &ids))
            }
            ir::Expr::ExtractDyn(x, i) => {
                let base = self.val(*x)?;
                computed(self.dynamic_part(base, self.ty_of(*x), *i)?)
            }
            ir::Expr::Splat(x, n) => {
                let ty = self.w.ty(t, false)?;
                let id = self.val(*x)?;
                if self.is_const(*x) {
                    Value { id: self.w.splat_const(ty, id, u32::from(*n)), constant: true }
                } else {
                    computed(self.op(Op::CompositeConstruct, ty, &vec![id; usize::from(*n)]))
                }
            }
            ir::Expr::Swizzle(x, comps) => {
                let ty = self.w.ty(t, false)?;
                let id = self.val(*x)?;
                let mut operands = vec![id, id];
                operands.extend(comps.iter().map(|c| u32::from(*c)));
                computed(self.op(Op::VectorShuffle, ty, &operands))
            }
            ir::Expr::Convert(x, s) => computed(self.convert(*x, *s, t, true)?),
            ir::Expr::Bitcast(x, s) => computed(self.convert(*x, *s, t, false)?),
            ir::Expr::Select { cond, if_true, if_false } => {
                let ty = self.w.ty(t, false)?;
                let mut c = self.val(*cond)?;
                let (a, b) = (self.val(*if_true)?, self.val(*if_false)?);
                // A scalar condition picking between vectors: a vector of it.
                if let (ir::TypeDef::Scalar(ir::Scalar::Bool), ir::TypeDef::Vector(_, n)) =
                    (m.types.get(self.ty_of(*cond)), m.types.get(t))
                {
                    let bt = self.w.bool_ty();
                    let bv = self.w.vec_ty(bt, u32::from(*n));
                    c = self.op(Op::CompositeConstruct, bv, &vec![c; usize::from(*n)]);
                }
                computed(self.op(Op::Select, ty, &[c, a, b]))
            }
            ir::Expr::Unary(op, x) => {
                let ty = self.w.ty(t, false)?;
                let id = self.val(*x)?;
                let xt = self.ty_of(*x);
                let op = match (op, m.types.get(xt)) {
                    (ir::UnOp::Neg, ir::TypeDef::Matrix(..)) => {
                        let minus = self.w.const_f32(-1.0);
                        return Ok(computed(self.op(Op::MatrixTimesScalar, ty, &[id, minus])));
                    }
                    (ir::UnOp::Neg, _) if self.scalar_of(*x).is_some_and(|s| s.is_float()) => {
                        Op::FNegate
                    }
                    (ir::UnOp::Neg, _) => Op::SNegate,
                    (
                        ir::UnOp::Not,
                        ir::TypeDef::Scalar(ir::Scalar::Bool)
                        | ir::TypeDef::Vector(ir::Scalar::Bool, _),
                    ) => Op::LogicalNot,
                    (ir::UnOp::Not, _) => Op::Not,
                };
                computed(self.op(op, ty, &[id]))
            }
            ir::Expr::Binary(op, a, b) => computed(self.binary(*op, *a, *b, t)?),
            ir::Expr::Builtin(b, args) => computed(self.builtin(*b, args, t)?),
            ir::Expr::Host(..) => return Err("GPU code can't record GPU work".into()),
            ir::Expr::Mem(..) => return Err("GPU code can't use raw memory".into()),
            ir::Expr::Run(_) | ir::Expr::Addr(_) => {
                return Err("internal: a CPU-only expression in GPU code".into());
            }
        })
    }

    /// `x` made scalar `s` (`numeric`: a conversion of its value; else its bits), as naga wrote
    /// each: a float clamped into the integer's range before it's made one.
    fn convert(&mut self, x: ir::ValueId, s: ir::Scalar, t: ir::TypeId, numeric: bool) -> R<Word> {
        let ty = self.w.ty(t, false)?;
        let id = self.val(x)?;
        let from = self.scalar_of(x).ok_or("internal: a conversion of a non-number")?;
        let n = self.width_of(x);
        if from == s {
            return Ok(id);
        }
        if !numeric {
            return Ok(self.op(Op::Bitcast, ty, &[id]));
        }
        use ir::Scalar as S;
        Ok(match (from, s) {
            (S::Bool, _) => {
                let st = self.w.scalar_ty(s)?;
                let (one, zero) = match s {
                    S::F32 => (self.w.const_f32(1.0), self.w.const_f32(0.0)),
                    S::I32 => (self.w.const_i32(1), self.w.const_i32(0)),
                    S::U32 => (self.w.const_u32(1), self.w.const_u32(0)),
                    S::F16 => {
                        let h = self.w.type_id(TypeKey::Float(16));
                        (
                            self.w
                                .constant(ConstKey::Scalar(h, u32::from(half::f16::ONE.to_bits()))),
                            self.w.constant(ConstKey::Scalar(h, 0)),
                        )
                    }
                    _ => return Err(format!("internal: bool to {s:?}")),
                };
                let _ = st;
                let (one, zero) = (self.w.splat_const(ty, one, n), self.w.splat_const(ty, zero, n));
                self.op(Op::Select, ty, &[id, one, zero])
            }
            (_, S::Bool) => {
                let xt = self.w.ty(self.ty_of(x), false)?;
                let (op, zero) = match from {
                    S::F32 => (Op::FUnordNotEqual, self.w.const_f32(0.0)),
                    S::I32 => (Op::INotEqual, self.w.const_i32(0)),
                    _ => (Op::INotEqual, self.w.const_u32(0)),
                };
                let zero = self.w.splat_const(xt, zero, n);
                self.op(op, ty, &[id, zero])
            }
            // A value its own code bounds inside the integer's range (`wrela_ir::ranges`) converts
            // as it is: the clamp would change nothing (above -1, truncation gives 0 too).
            (S::F32 | S::F16, S::I32 | S::U32)
                if self.ranges.within(
                    x,
                    if s == S::I32 { -2147483648.0 } else { -0.999 },
                    if s == S::I32 { 2147483520.0 } else { 4294967040.0 },
                ) =>
            {
                let op = if s == S::I32 { Op::ConvertFToS } else { Op::ConvertFToU };
                self.op(op, ty, &[id])
            }
            (S::F32 | S::F16, S::I32 | S::U32) => {
                let xt = self.w.ty(self.ty_of(x), false)?;
                let (lo, hi) = if s == S::I32 {
                    (-2147483648.0f32, 2147483520.0f32)
                } else {
                    (0.0f32, 4294967040.0f32)
                };
                let (lo, hi) = if from == S::F16 {
                    let h = self.w.type_id(TypeKey::Float(16));
                    let lo = if s == S::I32 { half::f16::MIN } else { half::f16::ZERO };
                    (
                        self.w.constant(ConstKey::Scalar(h, u32::from(lo.to_bits()))),
                        self.w.constant(ConstKey::Scalar(h, u32::from(half::f16::MAX.to_bits()))),
                    )
                } else {
                    (self.w.const_f32(lo), self.w.const_f32(hi))
                };
                let (lo, hi) = (self.w.splat_const(xt, lo, n), self.w.splat_const(xt, hi, n));
                let clamped = self.ext(GLOp::FClamp, xt, &[id, lo, hi]);
                let op = if s == S::I32 { Op::ConvertFToS } else { Op::ConvertFToU };
                self.op(op, ty, &[clamped])
            }
            (S::F32, S::F16) | (S::F16, S::F32) => self.op(Op::FConvert, ty, &[id]),
            (S::I32, S::F32 | S::F16) => self.op(Op::ConvertSToF, ty, &[id]),
            (S::U32, S::F32 | S::F16) => self.op(Op::ConvertUToF, ty, &[id]),
            _ => self.op(Op::Bitcast, ty, &[id]),
        })
    }

    fn binary(&mut self, op: ir::BinOp, a: ir::ValueId, b: ir::ValueId, t: ir::TypeId) -> R<Word> {
        use ir::BinOp as B;
        let m = self.w.m;
        let ty = self.w.ty(t, false)?;
        let (l, r) = (self.val(a)?, self.val(b)?);
        let (ta, tb) = (m.types.get(self.ty_of(a)).clone(), m.types.get(self.ty_of(b)).clone());
        let scalar = self.scalar_of(a);
        let float = scalar.is_some_and(|s| s.is_float());
        let signed = scalar == Some(ir::Scalar::I32);
        let boolean = scalar == Some(ir::Scalar::Bool);
        let is_matrix = |d: &ir::TypeDef| matches!(d, ir::TypeDef::Matrix(..));
        // Matrices: by columns, or the products SPIR-V has.
        if is_matrix(&ta) || is_matrix(&tb) {
            return Ok(match (op, &ta, &tb) {
                (B::Mul, ir::TypeDef::Matrix(..), ir::TypeDef::Matrix(..)) => {
                    self.op(Op::MatrixTimesMatrix, ty, &[l, r])
                }
                (B::Mul, ir::TypeDef::Matrix(..), ir::TypeDef::Vector(..)) => {
                    self.op(Op::MatrixTimesVector, ty, &[l, r])
                }
                (B::Mul, ir::TypeDef::Vector(..), ir::TypeDef::Matrix(..)) => {
                    self.op(Op::VectorTimesMatrix, ty, &[l, r])
                }
                (B::Mul, ir::TypeDef::Matrix(..), ir::TypeDef::Scalar(_)) => {
                    self.op(Op::MatrixTimesScalar, ty, &[l, r])
                }
                (B::Mul, ir::TypeDef::Scalar(_), ir::TypeDef::Matrix(..)) => {
                    self.op(Op::MatrixTimesScalar, ty, &[r, l])
                }
                (B::Add | B::Sub, ir::TypeDef::Matrix(c, rows), ir::TypeDef::Matrix(..)) => {
                    let f = self.w.f32_ty();
                    let col = self.w.vec_ty(f, u32::from(*rows));
                    let fop = if op == B::Add { Op::FAdd } else { Op::FSub };
                    let mut cols = Vec::new();
                    for k in 0..u32::from(*c) {
                        let x = self.op(Op::CompositeExtract, col, &[l, k]);
                        let y = self.op(Op::CompositeExtract, col, &[r, k]);
                        cols.push(self.op(fop, col, &[x, y]));
                    }
                    self.op(Op::CompositeConstruct, ty, &cols)
                }
                _ => return Err(format!("internal: {op:?} of matrices")),
            });
        }
        // Integer division and remainder, guarded as WGSL defines them: by 0 gives the
        // dividend (and 0 for a remainder), the most negative by -1 gives itself (and 0).
        if !float && matches!(op, B::Div | B::Rem) {
            let n = self.width_of(a);
            let guarded = self.w.checks.int_div || (signed && op == B::Rem);
            let divisor = if guarded && !self.w.proofs.safe_divisor(self.f, b) {
                self.safe_divisor(l, r, signed, n)?
            } else {
                r
            };
            return Ok(match (op, signed) {
                (B::Div, true) => self.op(Op::SDiv, ty, &[l, divisor]),
                (B::Div, false) => self.op(Op::UDiv, ty, &[l, divisor]),
                (_, true) => {
                    let q = self.op(Op::SDiv, ty, &[l, divisor]);
                    let p = self.op(Op::IMul, ty, &[q, divisor]);
                    self.op(Op::ISub, ty, &[l, p])
                }
                (_, false) => self.op(Op::UMod, ty, &[l, divisor]),
            });
        }
        let sop = match op {
            B::Add | B::WrappingAdd => {
                if float {
                    Op::FAdd
                } else {
                    Op::IAdd
                }
            }
            B::Sub | B::WrappingSub => {
                if float {
                    Op::FSub
                } else {
                    Op::ISub
                }
            }
            B::Mul | B::WrappingMul => {
                if float {
                    Op::FMul
                } else {
                    Op::IMul
                }
            }
            B::Div => Op::FDiv,
            B::Rem => Op::FRem,
            B::Shl => Op::ShiftLeftLogical,
            B::Shr if signed => Op::ShiftRightArithmetic,
            B::Shr => Op::ShiftRightLogical,
            B::BitAnd | B::And if boolean => Op::LogicalAnd,
            B::BitAnd | B::And => Op::BitwiseAnd,
            B::BitOr | B::Or if boolean => Op::LogicalOr,
            B::BitOr | B::Or => Op::BitwiseOr,
            B::BitXor if boolean => Op::LogicalNotEqual,
            B::BitXor => Op::BitwiseXor,
            B::Eq if boolean => Op::LogicalEqual,
            B::Eq if float => Op::FOrdEqual,
            B::Eq => Op::IEqual,
            B::Ne if boolean => Op::LogicalNotEqual,
            // Unordered: a NaN is unequal to everything, itself included (IEEE 754, WGSL).
            B::Ne if float => Op::FUnordNotEqual,
            B::Ne => Op::INotEqual,
            B::Lt if float => Op::FOrdLessThan,
            B::Lt if signed => Op::SLessThan,
            B::Lt => Op::ULessThan,
            B::Le if float => Op::FOrdLessThanEqual,
            B::Le if signed => Op::SLessThanEqual,
            B::Le => Op::ULessThanEqual,
            B::Gt if float => Op::FOrdGreaterThan,
            B::Gt if signed => Op::SGreaterThan,
            B::Gt => Op::UGreaterThan,
            B::Ge if float => Op::FOrdGreaterThanEqual,
            B::Ge if signed => Op::SGreaterThanEqual,
            B::Ge => Op::UGreaterThanEqual,
        };
        Ok(self.op(sop, ty, &[l, r]))
    }

    /// `rhs`, or 1 where dividing `lhs` by it isn't defined: by 0, or (signed) the most negative
    /// by -1.
    fn safe_divisor(&mut self, lhs: Word, rhs: Word, signed: bool, n: u32) -> R<Word> {
        let st = if signed { self.w.i32_ty() } else { self.w.u32_ty() };
        let vt = if n > 1 { self.w.vec_ty(st, n) } else { st };
        let bt = self.w.bool_ty();
        let bvt = if n > 1 { self.w.vec_ty(bt, n) } else { bt };
        let (zero, one) = if signed {
            (self.w.const_i32(0), self.w.const_i32(1))
        } else {
            (self.w.const_u32(0), self.w.const_u32(1))
        };
        let (zero, one) = (self.w.splat_const(vt, zero, n), self.w.splat_const(vt, one, n));
        let by_zero = self.op(Op::IEqual, bvt, &[rhs, zero]);
        let bad = if signed {
            let min = self.w.const_i32(i32::MIN);
            let minus = self.w.const_i32(-1);
            let (min, minus) = (self.w.splat_const(vt, min, n), self.w.splat_const(vt, minus, n));
            let is_min = self.op(Op::IEqual, bvt, &[lhs, min]);
            let is_minus = self.op(Op::IEqual, bvt, &[rhs, minus]);
            let overflow = self.op(Op::LogicalAnd, bvt, &[is_min, is_minus]);
            self.op(Op::LogicalOr, bvt, &[overflow, by_zero])
        } else {
            by_zero
        };
        Ok(self.op(Op::Select, vt, &[bad, one, rhs]))
    }

    fn builtin(&mut self, b: ir::Builtin, args: &[ir::ValueId], t: ir::TypeId) -> R<Word> {
        use ir::Builtin as B;
        let m = self.w.m;
        let ty = self.w.ty(t, false)?;
        let ids = args.iter().map(|a| self.val(*a)).collect::<R<Vec<_>>>()?;
        let a0 = args.first().copied().ok_or("internal: a builtin with no argument")?;
        let at = self.w.ty(self.ty_of(a0), false)?;
        let n = self.width_of(a0);
        let kind = self.scalar_of(a0);
        let float = kind.is_some_and(|s| s.is_float());
        let signed = kind == Some(ir::Scalar::I32);
        let derivative = |fw: &mut Self, op: Op| {
            fw.w.capability(Capability::DerivativeControl);
            fw.op(op, ty, &[ids[0]])
        };
        Ok(match b {
            B::Dpdx => derivative(self, Op::DPdxFine),
            B::Dpdy => derivative(self, Op::DPdyFine),
            B::Fwidth => derivative(self, Op::FwidthFine),
            B::AllEqual => {
                let bt = self.w.bool_ty();
                let bv = self.w.vec_ty(bt, n);
                let eq = if float {
                    Op::FOrdEqual
                } else if kind == Some(ir::Scalar::Bool) {
                    Op::LogicalEqual
                } else {
                    Op::IEqual
                };
                let e = self.op(eq, bv, &[ids[0], ids[1]]);
                self.op(Op::All, ty, &[e])
            }
            B::CountOnes => {
                let c = self.op(Op::BitCount, at, &[ids[0]]);
                if signed { self.op(Op::Bitcast, ty, &[c]) } else { c }
            }
            B::LeadingZeros => {
                let s = if signed { self.w.const_i32(31) } else { self.w.const_u32(31) };
                let s = self.w.splat_const(at, s, n);
                let msb = self.ext(GLOp::FindUMsb, at, &[ids[0]]);
                let c = self.op(Op::ISub, at, &[s, msb]);
                if signed { self.op(Op::Bitcast, ty, &[c]) } else { c }
            }
            B::TrailingZeros => {
                let s = if signed { self.w.const_i32(32) } else { self.w.const_u32(32) };
                let s = self.w.splat_const(at, s, n);
                let lsb = self.ext(GLOp::FindILsb, at, &[ids[0]]);
                let c = self.ext(GLOp::UMin, at, &[s, lsb]);
                if signed { self.op(Op::Bitcast, ty, &[c]) } else { c }
            }
            B::Abs if float => self.ext(GLOp::FAbs, ty, &ids),
            B::Abs if signed => self.ext(GLOp::SAbs, ty, &ids),
            B::Abs => self.op(Op::CopyObject, ty, &ids),
            B::Min if float => self.ext(GLOp::FMin, ty, &ids),
            B::Min if signed => self.ext(GLOp::SMin, ty, &ids),
            B::Min => self.ext(GLOp::UMin, ty, &ids),
            B::Max if float => self.ext(GLOp::FMax, ty, &ids),
            B::Max if signed => self.ext(GLOp::SMax, ty, &ids),
            B::Max => self.ext(GLOp::UMax, ty, &ids),
            B::Clamp if float => self.ext(GLOp::FClamp, ty, &ids),
            B::Clamp => {
                let (lo, hi) =
                    if signed { (GLOp::SMax, GLOp::SMin) } else { (GLOp::UMax, GLOp::UMin) };
                let x = self.ext(lo, ty, &[ids[0], ids[1]]);
                self.ext(hi, ty, &[x, ids[2]])
            }
            B::Saturate => {
                let (zero, one) = if kind == Some(ir::Scalar::F16) {
                    let h = self.w.type_id(TypeKey::Float(16));
                    (
                        self.w.constant(ConstKey::Scalar(h, 0)),
                        self.w.constant(ConstKey::Scalar(h, u32::from(half::f16::ONE.to_bits()))),
                    )
                } else {
                    (self.w.const_f32(0.0), self.w.const_f32(1.0))
                };
                let (zero, one) = (self.w.splat_const(ty, zero, n), self.w.splat_const(ty, one, n));
                self.ext(GLOp::FClamp, ty, &[ids[0], zero, one])
            }
            B::Sign if float => self.ext(GLOp::FSign, ty, &ids),
            B::Sign => self.ext(GLOp::SSign, ty, &ids),
            B::Mix => {
                // A scalar selector between vectors: a vector of it.
                let selector_n = self.width_of(args[2]);
                let mut operands = ids.clone();
                if n > 1 && selector_n == 1 {
                    operands[2] = self.op(Op::CompositeConstruct, ty, &vec![ids[2]; n as usize]);
                }
                self.ext(GLOp::FMix, ty, &operands)
            }
            B::Dot if float => self.op(Op::Dot, ty, &ids),
            B::Dot => {
                // Integers: the sum of the components' products, naga's.
                let mut sum = None;
                for k in 0..n {
                    let x = self.op(Op::CompositeExtract, ty, &[ids[0], k]);
                    let y = self.op(Op::CompositeExtract, ty, &[ids[1], k]);
                    let p = self.op(Op::IMul, ty, &[x, y]);
                    sum = Some(match sum {
                        None => p,
                        Some(s) => self.op(Op::IAdd, ty, &[s, p]),
                    });
                }
                sum.ok_or("internal: a dot of no components")?
            }
            _ => {
                let gl = match b {
                    B::Sqrt => GLOp::Sqrt,
                    B::InverseSqrt => GLOp::InverseSqrt,
                    B::Sin => GLOp::Sin,
                    B::Cos => GLOp::Cos,
                    B::Tan => GLOp::Tan,
                    B::Asin => GLOp::Asin,
                    B::Acos => GLOp::Acos,
                    B::Atan => GLOp::Atan,
                    B::Atan2 => GLOp::Atan2,
                    B::Exp => GLOp::Exp,
                    B::Exp2 => GLOp::Exp2,
                    B::Log => GLOp::Log,
                    B::Log2 => GLOp::Log2,
                    B::Sinh => GLOp::Sinh,
                    B::Cosh => GLOp::Cosh,
                    B::Tanh => GLOp::Tanh,
                    B::Pow => GLOp::Pow,
                    B::Floor => GLOp::Floor,
                    B::Ceil => GLOp::Ceil,
                    B::Round => GLOp::RoundEven,
                    B::Trunc => GLOp::Trunc,
                    B::Fract => GLOp::Fract,
                    B::Step => GLOp::Step,
                    B::Smoothstep => GLOp::SmoothStep,
                    B::Length => GLOp::Length,
                    B::Distance => GLOp::Distance,
                    B::Cross => GLOp::Cross,
                    B::Normalize => GLOp::Normalize,
                    _ => return Err(format!("internal: the builtin {b:?}")),
                };
                let _ = m;
                self.ext(gl, ty, &ids)
            }
        })
    }

    /// An atomic operation (`ir::Expr::Atomic`); with `want`, its value as that type (a
    /// compare-exchange's: a struct of the old value and whether it stored).
    fn atomic(
        &mut self,
        op: ir::AtomicOp,
        p: &ir::Place,
        args: &[ir::ValueId],
        want: Option<ir::TypeId>,
    ) -> R<Option<Word>> {
        use ir::AtomicOp as A;
        let m = self.w.m;
        let ptr = self.pointer(p)?;
        let ir::TypeDef::Atomic(s) = *m.types.get(ptr.ty) else {
            return Err("internal: an atomic operation on something that isn't atomic".into());
        };
        let st = self.w.scalar_ty(s)?;
        let (scope, semantics) = self.atomic_scope(ptr.class);
        if op == A::Load {
            return Ok(Some(self.op(Op::AtomicLoad, st, &[ptr.id, scope, semantics])));
        }
        let first = self.val(*args.first().ok_or("internal: an atomic without an operand")?)?;
        if op == A::Store {
            self.emit(Op::AtomicStore, &[ptr.id, scope, semantics, first]);
            return Ok(None);
        }
        let signed = s == ir::Scalar::I32;
        if op == A::CompareExchange {
            let value = self
                .val(*args.get(1).ok_or("internal: a compare-exchange without its new value")?)?;
            let old = self.op(
                Op::AtomicCompareExchange,
                st,
                &[ptr.id, scope, semantics, semantics, value, first],
            );
            let Some(t) = want else { return Ok(Some(old)) };
            let bt = self.w.bool_ty();
            let mut stored = self.op(Op::IEqual, bt, &[old, first]);
            let types = &m.types;
            if types.field(t, 1).and_then(|b| types.as_scalar(b)) == Some(ir::Scalar::U32) {
                let (one, zero) = (self.w.const_u32(1), self.w.const_u32(0));
                let u = self.w.u32_ty();
                stored = self.op(Op::Select, u, &[stored, one, zero]);
            }
            let ty = self.w.ty(t, false)?;
            return Ok(Some(self.op(Op::CompositeConstruct, ty, &[old, stored])));
        }
        let sop = match op {
            A::Add => Op::AtomicIAdd,
            A::Sub => Op::AtomicISub,
            A::Min if signed => Op::AtomicSMin,
            A::Min => Op::AtomicUMin,
            A::Max if signed => Op::AtomicSMax,
            A::Max => Op::AtomicUMax,
            A::And => Op::AtomicAnd,
            A::Or => Op::AtomicOr,
            A::Xor => Op::AtomicXor,
            A::Exchange => Op::AtomicExchange,
            A::Load | A::Store | A::CompareExchange => unreachable!("handled above"),
        };
        Ok(Some(self.op(sop, st, &[ptr.id, scope, semantics, first])))
    }

    /// A subgroup operation's value.
    fn subgroup(&mut self, op: ir::SubgroupOp, args: &[ir::ValueId], t: ir::TypeId) -> R<Word> {
        use ir::SubgroupOp as S;
        self.w.capability(Capability::GroupNonUniform);
        let ty = self.w.ty(t, false)?;
        let a0 = *args.first().ok_or("internal: a subgroup operation without its operand")?;
        let x = self.val(a0)?;
        let kind = self.scalar_of(a0);
        let float = kind.is_some_and(|s| s.is_float());
        let signed = kind == Some(ir::Scalar::I32);
        let boolean = kind == Some(ir::Scalar::Bool);
        let scope = self.w.const_u32(spirv::Scope::Subgroup as u32);
        let arithmetic = |fw: &mut Self, op: Op, group: spirv::GroupOperation| {
            fw.w.capability(Capability::GroupNonUniformArithmetic);
            fw.op(op, ty, &[scope, group as u32, x])
        };
        use spirv::GroupOperation as G;
        Ok(match op {
            S::Add => arithmetic(
                self,
                if float { Op::GroupNonUniformFAdd } else { Op::GroupNonUniformIAdd },
                G::Reduce,
            ),
            S::InclusiveAdd => arithmetic(
                self,
                if float { Op::GroupNonUniformFAdd } else { Op::GroupNonUniformIAdd },
                G::InclusiveScan,
            ),
            S::ExclusiveAdd => arithmetic(
                self,
                if float { Op::GroupNonUniformFAdd } else { Op::GroupNonUniformIAdd },
                G::ExclusiveScan,
            ),
            S::Min => arithmetic(
                self,
                if float {
                    Op::GroupNonUniformFMin
                } else if signed {
                    Op::GroupNonUniformSMin
                } else {
                    Op::GroupNonUniformUMin
                },
                G::Reduce,
            ),
            S::Max => arithmetic(
                self,
                if float {
                    Op::GroupNonUniformFMax
                } else if signed {
                    Op::GroupNonUniformSMax
                } else {
                    Op::GroupNonUniformUMax
                },
                G::Reduce,
            ),
            S::And => arithmetic(
                self,
                if boolean { Op::GroupNonUniformLogicalAnd } else { Op::GroupNonUniformBitwiseAnd },
                G::Reduce,
            ),
            S::Or => arithmetic(
                self,
                if boolean { Op::GroupNonUniformLogicalOr } else { Op::GroupNonUniformBitwiseOr },
                G::Reduce,
            ),
            S::Xor => arithmetic(
                self,
                if boolean { Op::GroupNonUniformLogicalXor } else { Op::GroupNonUniformBitwiseXor },
                G::Reduce,
            ),
            S::Any => {
                self.w.capability(Capability::GroupNonUniformVote);
                self.op(Op::GroupNonUniformAny, ty, &[scope, x])
            }
            S::All => {
                self.w.capability(Capability::GroupNonUniformVote);
                self.op(Op::GroupNonUniformAll, ty, &[scope, x])
            }
            S::Ballot => {
                self.w.capability(Capability::GroupNonUniformBallot);
                self.op(Op::GroupNonUniformBallot, ty, &[scope, x])
            }
            S::BroadcastFirst => {
                self.w.capability(Capability::GroupNonUniformBallot);
                self.op(Op::GroupNonUniformBroadcastFirst, ty, &[scope, x])
            }
            S::Shuffle => {
                self.w.capability(Capability::GroupNonUniformShuffle);
                let from =
                    self.val(*args.get(1).ok_or("internal: a shuffle without its source")?)?;
                self.op(Op::GroupNonUniformShuffle, ty, &[scope, x, from])
            }
        })
    }

    /// Texture or sampler `r`'s variable: its element of the device's table where the table
    /// holds it.
    fn texture_var(&mut self, r: usize) -> R<Word> {
        if let Some(&e) = self.elements.get(&r) {
            return Ok(e);
        }
        Ok(self.w.resource(r)?.0)
    }

    /// A texture's image, loaded from its variable, and its type.
    fn image(&mut self, tex: ir::ResourceId) -> R<(Word, Word)> {
        let var = self.texture_var(tex.index())?;
        let res = &self.w.m.resources[tex.index()];
        let ty = match res.kind {
            ir::ResourceKind::Texture { depth, dim, format } => {
                let sampled = if ir::texel_scalar(format) == ir::Scalar::U32 && !depth {
                    self.w.u32_ty()
                } else {
                    self.w.f32_ty()
                };
                let (d, arrayed) = image_dim(dim);
                self.w.type_id(TypeKey::Image { sampled, dim: d, depth, arrayed, storage: None })
            }
            ir::ResourceKind::StorageTexture { .. } => {
                // Declared by `resource`; its type is the variable's pointee.
                return Err("internal: a storage texture read".into());
            }
            _ => return Err("internal: an image of something that isn't a texture".into()),
        };
        Ok((self.op(Op::Load, ty, &[var]), ty))
    }

    /// A texture read (`ir::Expr::Texture`) as a value of IR type `t`.
    fn texture(
        &mut self,
        op: ir::TextureOp,
        tex: ir::ResourceId,
        sampler: Option<ir::ResourceId>,
        args: &[ir::ValueId],
        t: ir::TypeId,
    ) -> R<Word> {
        use ir::TextureOp as T;
        self.w.capability(Capability::ImageQuery);
        let m = self.w.m;
        let ty = self.w.ty(t, false)?;
        let (dim, depth) = match m.resources[tex.index()].kind {
            ir::ResourceKind::Texture { dim, depth, .. } => (dim, depth),
            _ => return Err("internal: a texture read of something that isn't a texture".into()),
        };
        let (image, image_ty) = self.image(tex)?;
        let arg = |fw: &Self, i: usize| -> R<Word> {
            fw.val(*args.get(i).ok_or("internal: a texture read without its arguments")?)
        };
        let u = self.w.u32_ty();
        let i32t = self.w.i32_ty();
        let f = self.w.f32_ty();
        let f4 = self.w.vec_ty(f, 4);
        let layer_at = if op == T::Load { dim.texel() } else { 1 };
        let layer = if dim.arrayed() && !op.is_query() { Some(arg(self, layer_at)?) } else { None };
        let rest = layer_at + usize::from(layer.is_some());
        Ok(match op {
            T::Sample
            | T::SampleLevel
            | T::SampleGrad
            | T::SampleBias
            | T::SampleCompare
            | T::SampleCompareLevel
            | T::Gather(_)
            | T::GatherCompare => {
                let s = sampler.ok_or("internal: sampling without a sampler")?;
                let svar = self.texture_var(s.index())?;
                let sampler_ty = self.w.type_id(TypeKey::Sampler);
                let smp = self.op(Op::Load, sampler_ty, &[svar]);
                let si_ty = self.w.type_id(TypeKey::SampledImage(image_ty));
                let si = self.op(Op::SampledImage, si_ty, &[image, smp]);
                let mut coords = arg(self, 0)?;
                if let Some(l) = layer {
                    // The layer as a float, after the coordinates.
                    let lf = self.op(Op::ConvertUToF, f, &[l]);
                    let n = u32::from(dim.coordinates()) + 1;
                    let ct = self.w.vec_ty(f, n);
                    coords = self.op(Op::CompositeConstruct, ct, &[coords, lf]);
                }
                let dref = match op {
                    T::SampleCompare | T::SampleCompareLevel | T::GatherCompare => {
                        Some(arg(self, rest)?)
                    }
                    _ => None,
                };
                // A depth texture read without a reference gives four of its depth: the first.
                let sub = depth && dref.is_none() && !matches!(op, T::Gather(_));
                let result = if sub || dref.is_some() && !matches!(op, T::GatherCompare) {
                    if sub { f4 } else { f }
                } else {
                    ty
                };
                let id = match op {
                    T::Gather(c) => {
                        let comp = self.w.const_u32(u32::from(c));
                        self.op(Op::ImageGather, result, &[si, coords, comp])
                    }
                    T::GatherCompare => {
                        self.op(Op::ImageDrefGather, result, &[si, coords, dref.unwrap_or(0)])
                    }
                    T::Sample => self.op(Op::ImageSampleImplicitLod, result, &[si, coords]),
                    T::SampleBias => {
                        let bias = arg(self, rest)?;
                        self.op(Op::ImageSampleImplicitLod, result, &[si, coords, 1, bias])
                    }
                    T::SampleLevel => {
                        let level = arg(self, rest)?;
                        self.op(Op::ImageSampleExplicitLod, result, &[si, coords, 2, level])
                    }
                    T::SampleGrad => {
                        let (dx, dy) = (arg(self, rest)?, arg(self, rest + 1)?);
                        self.op(Op::ImageSampleExplicitLod, result, &[si, coords, 4, dx, dy])
                    }
                    T::SampleCompare => self.op(
                        Op::ImageSampleDrefImplicitLod,
                        result,
                        &[si, coords, dref.unwrap_or(0)],
                    ),
                    T::SampleCompareLevel => {
                        let zero = self.w.const_f32(0.0);
                        self.op(
                            Op::ImageSampleDrefExplicitLod,
                            result,
                            &[si, coords, dref.unwrap_or(0), 2, zero],
                        )
                    }
                    _ => unreachable!("sampling ops only"),
                };
                if sub { self.op(Op::CompositeExtract, ty, &[id, 0]) } else { id }
            }
            T::Load => {
                let n = dim.texel();
                let mut coords = Vec::new();
                for i in 0..n {
                    let c = arg(self, i)?;
                    coords.push(self.op(Op::Bitcast, i32t, &[c]));
                }
                if let Some(l) = layer {
                    coords.push(self.op(Op::Bitcast, i32t, &[l]));
                }
                let cn = coords.len() as u32;
                let ivec = self.w.vec_ty(i32t, cn);
                let mut coord = self.op(Op::CompositeConstruct, ivec, &coords);
                let level = arg(self, rest)?;
                let mut level = self.op(Op::Bitcast, i32t, &[level]);
                if self.w.checks.bounds {
                    // The level, then the texel (and the layer) clamped into what's there.
                    let one = self.w.const_i32(1);
                    let levels = self.op(Op::ImageQueryLevels, i32t, &[image]);
                    let top = self.op(Op::ISub, i32t, &[levels, one]);
                    level = self.ext(GLOp::UMin, i32t, &[level, top]);
                    let size = self.op(Op::ImageQuerySizeLod, ivec, &[image, level]);
                    let ones = self.w.splat_const(ivec, one, cn);
                    let limit = self.op(Op::ISub, ivec, &[size, ones]);
                    coord = self.ext(GLOp::UMin, ivec, &[coord, limit]);
                }
                let result = if depth { f4 } else { ty };
                let id = self.op(Op::ImageFetch, result, &[image, coord, 2, level]);
                if depth { self.op(Op::CompositeExtract, ty, &[id, 0]) } else { id }
            }
            T::Width | T::Height | T::Depth | T::Layers => {
                let n = match dim {
                    ir::TexDim::D3 | ir::TexDim::D2Array => 3,
                    _ => 2,
                };
                let uv = self.w.vec_ty(u, n);
                let zero = self.w.const_u32(0);
                let size = self.op(Op::ImageQuerySizeLod, uv, &[image, zero]);
                let k = match op {
                    T::Width => 0,
                    T::Height => 1,
                    _ => 2,
                };
                self.op(Op::CompositeExtract, ty, &[size, k])
            }
            T::Levels => self.op(Op::ImageQueryLevels, ty, &[image]),
        })
    }

    /// A texel's write (`ir::Expr::TextureStore`).
    fn texture_store(&mut self, tex: ir::ResourceId, args: &[ir::ValueId]) -> R<()> {
        let m = self.w.m;
        let var = self.texture_var(tex.index())?;
        let (dim, format) = match m.resources[tex.index()].kind {
            ir::ResourceKind::StorageTexture { dim, format } => (dim, format),
            _ => {
                return Err(
                    "internal: a texel write of something that isn't a storage texture".into()
                );
            }
        };
        let sampled = if ir::texel_scalar(format) == ir::Scalar::U32 {
            self.w.u32_ty()
        } else {
            self.w.f32_ty()
        };
        let storage = match format {
            ir::TextureFormat::Rgba8 => ImageFormat::Rgba8,
            ir::TextureFormat::Rgba16Float => ImageFormat::Rgba16f,
            ir::TextureFormat::R16Float => ImageFormat::R16f,
            ir::TextureFormat::R32Float => ImageFormat::R32f,
            ir::TextureFormat::R32Uint => ImageFormat::R32ui,
            ir::TextureFormat::Rg32Float => ImageFormat::Rg32f,
            ir::TextureFormat::Rgba32Float => ImageFormat::Rgba32f,
            ir::TextureFormat::Rg32Uint => ImageFormat::Rg32ui,
            f => return Err(format!("internal: kernels can't write a {f:?} texture")),
        };
        let (d, arrayed) = image_dim(dim);
        let ity = self.w.type_id(TypeKey::Image {
            sampled,
            dim: d,
            depth: false,
            arrayed,
            storage: Some(storage),
        });
        let image = self.op(Op::Load, ity, &[var]);
        let Some((&value, xs)) = args.split_last() else {
            return Err("internal: a texel's write without its arguments".into());
        };
        let i32t = self.w.i32_ty();
        let mut coords = Vec::new();
        for x in xs {
            let c = self.val(*x)?;
            coords.push(self.op(Op::Bitcast, i32t, &[c]));
        }
        let ivec = self.w.vec_ty(i32t, coords.len() as u32);
        let coord = self.op(Op::CompositeConstruct, ivec, &coords);
        let value = self.val(value)?;
        self.emit(Op::ImageWrite, &[image, coord, value]);
        Ok(())
    }
}
