//! The Metal back end, written from the IR directly: a GPU module of [`wrela_ir`] as Metal shading
//! language (MSL 4.0), each binding in the argument-table slot [`wrela_abi::metal`] gives it, or,
//! a texture or sampler the device's table holds ([`wrela_abi::table`]), read from that table at
//! its index in the block of indices: for the host to hand the GPU as it is.
//!
//! It computes what naga 30's MSL writer computed for the same module (`msl`, which it replaces):
//! the same operations in the same order, with the same helpers where MSL's own operators leave a
//! case undefined (naga's `naga_div`, `naga_mod`, `naga_neg`, `naga_abs`, `naga_f2i32`,
//! `naga_f2u32`), the same run-time checks ([`Checks`]: indexes clamped into range, loops given a
//! bound the compiler can see, integer division guarded) and workgroup memory zeroed before a
//! kernel's first barrier. naga's text was read as reference (docs/citations.md); nothing of it is
//! copied. Where naga's writer couldn't say something, this one says it directly:
//!
//! - **Buffer sizes.** Each runtime-sized buffer's bound length is read from the pipeline's own
//!   uniform buffer ([`wrela_abi::manifest::BufferSizes`]), which the host fills.
//! - **The table** (`constant ulong* _table [[buffer(30)]]`, the `MTLResourceID`s of every view
//!   and sampler) is read as an array of a one-member struct of the type a resource is, at its
//!   index in the block (`constant _Block& _blk [[buffer(29)]]`): an entry point makes each such
//!   resource a local, and passes it on to the functions it calls as it would a bound one.
//! - **Workgroup memory** is declared in the kernel's body, so Metal sizes it from its type.
//! - **Conservative depth.** A fragment shader's promised depth is Metal's
//!   `[[depth(greater)]]` or `[[depth(less)]]`.
//! - **The math mode is stated:** `#pragma METAL fp math_mode(relaxed)` and `fp contract(fast)`:
//!   fast math that keeps infinities and NaNs (language.md §11.3), not the `fast` a passthrough
//!   shader would inherit from `MTLCompileOptions`' defaults. A float compared with itself is
//!   `isnan`, which even relaxed math can't fold away.
//!
//! The text is written for a compiler, not a reader: one statement a line, no indentation. Each IR
//! value that reads memory, calls, has an effect, is used more than once or is a derivative is
//! bound to a name where the IR defines it (`vN`), so evaluation order is the IR's; any other is
//! written where it's used.

use crate::Checks;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use wrela_abi::manifest::{BufferSizes, Pipeline, UniformSpace};
use wrela_abi::metal::{self as slots, ShaderStage, Table};
use wrela_ir as ir;
use wrela_ir::layout::WgslLayouts;

type R<T> = Result<T, String>;

/// The MSL language version written: what Metal compiles a passthrough shader with on macOS 26
/// (its default, the newest).
pub const LANGUAGE: (u8, u8) = (4, 0);

/// A GPU module written as MSL.
#[derive(Clone, Debug)]
pub struct Msl {
    pub text: String,
    /// Each entry point's name in `text`, in the module's order.
    pub entry_points: Vec<String>,
}

/// What a fragment shader's depth promises (`std::gpu::WithDepthBehind`): never nearer than its
/// triangle's depth, so the GPU may test the triangle's depth before the shader runs. Which way
/// "nearer" is depends on the pipeline's depth test: `Greater` for a test that keeps less, `Less`
/// for one that keeps greater (reversed depth).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConservativeDepth {
    Greater,
    Less,
}

impl ConservativeDepth {
    fn qualifier(self) -> &'static str {
        match self {
            ConservativeDepth::Greater => "depth(greater)",
            ConservativeDepth::Less => "depth(less)",
        }
    }
}

/// The bindings of the module's runtime-sized buffers, in its order: what the sizes buffer
/// holds a length for, one `u32` each.
pub fn sized_bindings(m: &ir::Module) -> Vec<u32> {
    m.resources.iter().filter(|r| is_sized(r)).map(|r| r.binding).collect()
}

fn is_sized(r: &ir::Resource) -> bool {
    matches!(r.kind, ir::ResourceKind::StorageRead | ir::ResourceKind::StorageReadWrite)
}

/// The pipeline's sizes binding, if its shader reads sizes: one past every other binding.
pub fn buffer_sizes(m: &ir::Module, p: &Pipeline) -> Option<BufferSizes> {
    let buffers = sized_bindings(m);
    if buffers.is_empty() {
        return None;
    }
    let used = p
        .bindings
        .iter()
        .map(|b| b.binding)
        .chain(p.uniform.as_ref().map(|u| u.binding))
        .chain(p.debug_flag);
    let binding = used.max().map_or(0, |b| b + 1);
    Some(BufferSizes { binding, buffers })
}

/// Module `m` as MSL for pipeline `p`, whose manifest entry (bindings, uniform block, debug flag
/// and `sizes`, from [`buffer_sizes`]) says where its bindings go; with `depth` on its fragment
/// shader's depth, and `checks` written in.
pub fn write(
    m: &ir::Module,
    p: &Pipeline,
    depth: Option<ConservativeDepth>,
    checks: Checks,
) -> R<Msl> {
    let mut w = Writer::new(m, p, depth, checks)?;
    w.module()?;
    Ok(Msl { text: w.finish(), entry_points: w.entry_names })
}

// ---- names ---------------------------------------------------------------------------------------

/// An entry point's name: the IR's, with `_e` after it. Every name the writer makes has a letter
/// of its own before or after the IR's (`r0_`, `f3_`, `S5_`, `v12`), so none is one of C++'s or
/// Metal's (a function named `quad` is a redefinition of Metal's own); the manifest takes the
/// entry point's name from the shader.
fn entry_name(name: &str) -> String {
    format!("{}_e", ir::ident(name))
}

/// A name for a struct member, local, parameter or resource: a letter that says which, then
/// the IR's own name, so no name is a keyword or another's.
fn named(prefix: &str, name: &str, k: usize) -> String {
    format!("{prefix}{k}_{}", ir::ident(name))
}

// ---- types -----------------------------------------------------------------------------------------

/// A struct as MSL lays it: each member (its name, its type's text, packed or not) and before it
/// the bytes of padding that put it at the IR's offset.
struct StructDef {
    /// Members with a pad before them (by member index): a brace initializer gives each pad `{}`.
    pads: HashSet<usize>,
    members: Vec<String>,
    /// Members held as packed three-component vectors: read through a conversion.
    packed: HashSet<usize>,
}

fn scalar_name(s: ir::Scalar) -> R<&'static str> {
    Ok(match s {
        ir::Scalar::Bool => "bool",
        ir::Scalar::I32 => "int",
        ir::Scalar::U32 => "uint",
        ir::Scalar::F32 => "float",
        ir::Scalar::F16 => "half",
        other => return Err(format!("internal: `{}` in GPU code", other.name())),
    })
}

fn vector_name(s: ir::Scalar, n: u8) -> R<String> {
    Ok(format!("metal::{}{n}", scalar_name(s)?))
}

/// A type's size in MSL as naga sized it to place the next member (a `vec3` 12 bytes: what
/// follows a packed one starts there, and an unpacked one's own fourth component is added).
fn msl_size(types: &ir::Types, layouts: &mut WgslLayouts, t: ir::TypeId) -> u32 {
    match types.get(t) {
        ir::TypeDef::Scalar(ir::Scalar::Bool) => 1,
        ir::TypeDef::Scalar(s) | ir::TypeDef::Atomic(s) => s.bytes(),
        ir::TypeDef::Vector(ir::Scalar::Bool, n) => u32::from(*n),
        ir::TypeDef::Vector(s, n) => s.bytes() * u32::from(*n),
        ir::TypeDef::Matrix(c, r) => {
            let rows = if *r == 3 { 4 } else { u32::from(*r) };
            u32::from(*c) * rows * 4
        }
        _ => layouts.get(types, t).0.size,
    }
}

// ---- the writer ------------------------------------------------------------------------------------

/// A helper function the text calls, written once before the functions.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Helper {
    /// `naga_div` or `naga_mod` of a scalar or vector type's name.
    Div(String, bool),
    Mod(String, bool),
    /// `naga_neg` of a signed type and its unsigned twin.
    Neg(String, String),
    /// `naga_abs` of a signed type and its unsigned twin.
    Abs(String, String),
    /// `naga_f2i32` or `naga_f2u32` from a float type to an int type.
    F2I(String, String, bool),
    /// `naga_dot_<int type>` of an integer vector type and its size.
    Dot(String, String, u8),
    /// The compare-exchange of an atomic of this scalar, and the result struct's name.
    CompareExchange(String, String),
}

struct Writer<'m> {
    m: &'m ir::Module,
    p: &'m Pipeline,
    depth: Option<ConservativeDepth>,
    checks: Checks,
    /// What the compiler proves of the checks: the ones it makes unneeded are left out.
    proofs: wrela_ir::work::Proofs,
    layouts: WgslLayouts,
    /// Each type's name in MSL, made once.
    type_names: HashMap<ir::TypeId, String>,
    /// The structs written, by type.
    structs: HashMap<ir::TypeId, StructDef>,
    /// Type definitions, in an order where each follows what it uses.
    type_defs: String,
    helpers: BTreeSet<Helper>,
    /// The compare-exchange result structs, by their scalar's name.
    exchange_results: BTreeSet<String>,
    /// Constant tables at program scope.
    constants: String,
    /// The functions, helpers before the entry points.
    functions: String,
    /// Each resource's name.
    resource_names: Vec<String>,
    /// Each function's name.
    function_names: Vec<String>,
    /// The resources each function uses, itself or through the functions it calls.
    uses: Vec<BTreeSet<usize>>,
    /// Each table's name.
    data_names: HashMap<ir::DataId, String>,
    /// The element types of sized buffers whose array type is written.
    runtime_arrays: HashSet<ir::TypeId>,
    /// Each sized buffer's member in `_mslBufferSizes`, by resource.
    size_members: HashMap<usize, String>,
    entry_names: Vec<String>,
    /// The one-member struct each texture or sampler type is read from the device's table as,
    /// by type; and whether the block of indices' struct is written.
    table_types: HashMap<String, String>,
    block_written: bool,
    /// Each function's: whether some call to it, or to a function that calls it, is inside a
    /// loop (`repeated`).
    repeated: Vec<bool>,
}

/// For each function of `m`, whether it runs inside a loop: some call to it is in a loop, or in
/// a function that does.
fn repeated(m: &ir::Module) -> Vec<bool> {
    fn scan(b: &[ir::Stmt], depth: u32, calls: &mut Vec<(usize, bool)>) {
        for s in b {
            if let Some(ir::Expr::Call(g, _)) = s.expr() {
                calls.push((g.index(), depth > 0));
            }
            match s {
                ir::Stmt::If { then, else_, .. } => {
                    scan(then, depth, calls);
                    scan(else_, depth, calls);
                }
                ir::Stmt::Loop { body, continuing } => {
                    scan(body, depth + 1, calls);
                    scan(continuing, depth + 1, calls);
                }
                _ => {}
            }
        }
    }
    let calls: Vec<Vec<(usize, bool)>> = m
        .functions
        .iter()
        .map(|f| {
            let mut c = Vec::new();
            scan(&f.body, 0, &mut c);
            c
        })
        .collect();
    let mut out = vec![false; m.functions.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for (f, cs) in calls.iter().enumerate() {
            for &(g, in_loop) in cs {
                if g < out.len() && !out[g] && (in_loop || out[f]) {
                    out[g] = true;
                    changed = true;
                }
            }
        }
    }
    out
}

impl<'m> Writer<'m> {
    fn new(
        m: &'m ir::Module,
        p: &'m Pipeline,
        depth: Option<ConservativeDepth>,
        checks: Checks,
    ) -> R<Writer<'m>> {
        let resource_names =
            m.resources.iter().enumerate().map(|(k, r)| named("r", &r.name, k)).collect();
        let function_names =
            m.functions.iter().enumerate().map(|(k, f)| named("f", &f.name, k)).collect();
        let size_members = m
            .resources
            .iter()
            .enumerate()
            .filter(|(_, r)| is_sized(r))
            .enumerate()
            .map(|(k, (i, _))| (i, format!("size{k}")))
            .collect();
        let mut w = Writer {
            m,
            p,
            depth,
            checks,
            proofs: if checks == Checks::UNCHECKED {
                wrela_ir::work::Proofs::default()
            } else {
                wrela_ir::work::proofs(m)
            },
            layouts: WgslLayouts::default(),
            type_names: HashMap::new(),
            structs: HashMap::new(),
            type_defs: String::new(),
            helpers: BTreeSet::new(),
            exchange_results: BTreeSet::new(),
            constants: String::new(),
            functions: String::new(),
            resource_names,
            function_names,
            uses: Vec::new(),
            data_names: HashMap::new(),
            runtime_arrays: HashSet::new(),
            repeated: repeated(m),
            size_members,
            entry_names: Vec::new(),
            table_types: HashMap::new(),
            block_written: false,
        };
        w.uses = w.resource_uses();
        Ok(w)
    }

    /// The resources each function uses, directly or through its calls.
    fn resource_uses(&self) -> Vec<BTreeSet<usize>> {
        let m = self.m;
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
                        expr_resources(e, &mut used);
                    }
                });
                used
            })
            .collect();
        (0..m.functions.len())
            .map(|f| {
                let mut all = BTreeSet::new();
                for g in ir::visit::reachable(m, ir::FuncId(f as u32), &[]) {
                    all.extend(direct[g.index()].iter().copied());
                }
                all.extend(direct[f].iter().copied());
                all
            })
            .collect()
    }

    fn finish(&mut self) -> String {
        let mut out = String::with_capacity(
            self.type_defs.len() + self.constants.len() + self.functions.len() + 4096,
        );
        let (major, minor) = LANGUAGE;
        let _ = writeln!(out, "// language: metal{major}.{minor}");
        out.push_str("#include <metal_stdlib>\n#include <simd/simd.h>\n");
        // Relaxed: fast math that keeps infinities and NaNs (language.md §11.3); `fast` lets the
        // compiler assume there are none, and `safe` costs the opening's frame 91% on an M4.
        out.push_str("#pragma METAL fp math_mode(relaxed)\n");
        out.push_str("#pragma METAL fp contract(fast)\n");
        // `x == x` is how GPU code asks whether a float is a number: meant, not a slip.
        out.push_str("#pragma clang diagnostic ignored \"-Wtautological-compare\"\n");
        // A runtime-sized array is declared with one element; an index past it is meant.
        out.push_str("#pragma clang diagnostic ignored \"-Warray-bounds\"\n");
        out.push_str("using metal::uint;\n");
        if !self.size_members.is_empty() {
            let mut members: Vec<(&usize, &String)> = self.size_members.iter().collect();
            members.sort();
            out.push_str("struct _mslBufferSizes {\n");
            for (_, name) in members {
                let _ = writeln!(out, "uint {name};");
            }
            out.push_str("};\n");
        }
        out.push_str(&self.type_defs);
        for scalar in &self.exchange_results {
            let _ = writeln!(
                out,
                "struct exchange_{scalar} {{\n{scalar} old_value;\nbool exchanged;\n}};"
            );
        }
        out.push_str(&self.constants);
        for h in &self.helpers {
            write_helper(&mut out, h);
        }
        out.push_str(&self.functions);
        out
    }

    // ---- types ----

    fn ty(&mut self, t: ir::TypeId) -> R<String> {
        if let Some(n) = self.type_names.get(&t) {
            return Ok(n.clone());
        }
        let m = self.m;
        let name = match m.types.get(t).clone() {
            ir::TypeDef::Scalar(s) => scalar_name(s)?.to_string(),
            ir::TypeDef::Atomic(s) => format!("metal::atomic_{}", scalar_name(s)?),
            ir::TypeDef::Vector(s, n) => vector_name(s, n)?,
            ir::TypeDef::Matrix(c, r) => format!("metal::float{c}x{r}"),
            ir::TypeDef::Array(e, n) => {
                let elem = self.ty(e)?;
                let name = format!("A{}", t.0);
                let _ = writeln!(self.type_defs, "struct {name} {{\n{elem} inner[{n}];\n}};");
                name
            }
            ir::TypeDef::RuntimeArray(e) => return self.ty(e),
            ir::TypeDef::Struct { name, .. } | ir::TypeDef::Enum { name, .. } => {
                let name = format!("S{}_{}", t.0, short(&name));
                self.struct_def(t, &name)?;
                name
            }
            ir::TypeDef::Run(_) | ir::TypeDef::Ptr(_) => {
                return Err("internal: a CPU-only type in GPU code".into());
            }
        };
        self.type_names.insert(t, name.clone());
        Ok(name)
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

    /// Writes struct `t` (its members' types first), each member at the IR's offset: padding
    /// before a member that would sit earlier, and a three-component vector followed tightly by
    /// the next member held packed.
    fn struct_def(&mut self, t: ir::TypeId, name: &str) -> R<()> {
        let members = self.members(t)?;
        let mut types = Vec::with_capacity(members.len());
        for (_, mt) in &members {
            types.push(self.ty(*mt)?);
        }
        let (layout, offsets) = self.layouts.get(&self.m.types, t);
        let (span, offsets) = (layout.size, offsets.to_vec());
        let mut def =
            StructDef { pads: HashSet::new(), members: Vec::new(), packed: HashSet::new() };
        let mut text = format!("struct {name} {{\n");
        let mut last = 0;
        for (k, ((mname, mt), tyname)) in members.iter().zip(&types).enumerate() {
            let offset = offsets[k];
            if offset > last {
                def.pads.insert(k);
                let _ = writeln!(text, "char _pad{k}[{}];", offset - last);
            } else if offset < last {
                return Err(format!("internal: struct `{name}` overlaps at member {k}"));
            }
            let size = msl_size(&self.m.types, &mut self.layouts, *mt);
            last = offset + size;
            let next = offsets.get(k + 1).copied().unwrap_or(span);
            let member = named("m", mname, k);
            match self.m.types.get(*mt) {
                ir::TypeDef::Vector(s, 3) if next == last && s.bytes() >= 2 => {
                    def.packed.insert(k);
                    let _ = writeln!(text, "metal::packed_{}3 {member};", scalar_name(*s)?);
                }
                ir::TypeDef::Vector(s, 3) => {
                    let _ = writeln!(text, "{tyname} {member};");
                    last += s.bytes();
                }
                _ => {
                    let _ = writeln!(text, "{tyname} {member};");
                }
            }
            def.members.push(member);
        }
        if last < span {
            let _ = writeln!(text, "char _pad{}[{}];", members.len(), span - last);
        }
        text.push_str("};\n");
        self.type_defs.push_str(&text);
        self.structs.insert(t, def);
        Ok(())
    }

    /// The struct member that holds field `k` of a value of type `t`: for an enum, its tag, or
    /// the payload of variant `k - 1` among the variants that have one.
    fn member(&self, t: ir::TypeId, k: u32) -> u32 {
        match self.m.types.get(t) {
            ir::TypeDef::Enum { variants, .. } if k > 0 => {
                1 + variants[..k as usize - 1].iter().filter(|(_, p)| p.is_some()).count() as u32
            }
            _ => k,
        }
    }

    /// Member `k`'s name of struct `t`, and whether it's packed.
    fn member_name(&mut self, t: ir::TypeId, k: u32) -> R<(String, bool)> {
        self.ty(t)?;
        let def = self.structs.get(&t).ok_or("internal: a member of a type that isn't a struct")?;
        let name =
            def.members.get(k as usize).cloned().ok_or("internal: a struct's member is missing")?;
        Ok((name, def.packed.contains(&(k as usize))))
    }

    // ---- constants ----

    /// Table `d` at program scope, written once.
    fn data(&mut self, d: ir::DataId) -> R<String> {
        if let Some(n) = self.data_names.get(&d) {
            return Ok(n.clone());
        }
        let m = self.m;
        let data = m.data.get(d.index()).ok_or("internal: missing constant data")?;
        let ty = self.ty(data.ty)?;
        let value = self.const_value(data.ty, &data.value)?;
        let name = format!("T{}", self.data_names.len());
        let _ = writeln!(self.constants, "constant {ty} {name} = {value};");
        self.data_names.insert(d, name.clone());
        Ok(name)
    }

    fn const_value(&mut self, t: ir::TypeId, v: &ir::ConstValue) -> R<String> {
        let m = self.m;
        match (m.types.get(t).clone(), v) {
            (ir::TypeDef::Scalar(_), ir::ConstValue::Scalar(c)) => literal(c),
            (ir::TypeDef::Vector(..), ir::ConstValue::Parts(parts)) => {
                let ty = self.ty(t)?;
                let parts = parts
                    .iter()
                    .map(|p| match p {
                        ir::ConstValue::Scalar(c) => literal(c),
                        _ => Err("internal: a vector's part in a table".to_string()),
                    })
                    .collect::<R<Vec<_>>>()?;
                Ok(format!("{ty}({})", parts.join(", ")))
            }
            (ir::TypeDef::Array(e, _), ir::ConstValue::Parts(parts)) => {
                let ty = self.ty(t)?;
                let parts = parts.iter().map(|p| self.const_value(e, p)).collect::<R<Vec<_>>>()?;
                Ok(format!("{ty} {{{{{}}}}}", parts.join(", ")))
            }
            (ir::TypeDef::Struct { fields, .. }, ir::ConstValue::Parts(parts)) => {
                let ty = self.ty(t)?;
                let mut values = Vec::new();
                for ((_, ft), p) in fields.iter().zip(parts) {
                    values.push(self.const_value(*ft, p)?);
                }
                Ok(format!("{ty} {}", self.braced(t, &values)))
            }
            _ => Err("internal: a table's value doesn't fit its type".into()),
        }
    }

    /// A brace initializer of struct `t` from its members' values, each pad given `{}`.
    fn braced(&self, t: ir::TypeId, values: &[String]) -> String {
        let pads = self.structs.get(&t).map(|d| &d.pads);
        let mut out = String::from("{");
        for (k, v) in values.iter().enumerate() {
            if k > 0 {
                out.push_str(", ");
            }
            if pads.is_some_and(|p| p.contains(&k)) {
                out.push_str("{}, ");
            }
            out.push_str(v);
        }
        out.push('}');
        out
    }

    // ---- resources ----

    /// Resource `r` as an argument of a function: its type and name, and in an entry point its
    /// slot (`slot` given).
    fn resource_arg(&mut self, r: usize, slot: Option<&slots::Slot>) -> R<String> {
        let res = &self.m.resources[r];
        let name = self.resource_names[r].clone();
        let at =
            |table: &str| slot.map(|s| format!(" [[{table}({})]]", s.index)).unwrap_or_default();
        Ok(match res.kind {
            ir::ResourceKind::Uniform { storage } => {
                let ty = self.ty(res.ty)?;
                let space = match (storage, self.uniform_space()) {
                    (true, _) | (_, Some(UniformSpace::Storage)) => "device",
                    _ => "constant",
                };
                let constness = if space == "device" { " const" } else { "" };
                format!("{space} {ty}{constness}& {name}{}", at("buffer"))
            }
            // A sized buffer as a reference to an array of one element, indexed past it: as naga
            // passed one (a pointer reads the same, but Metal's fast math then meets the
            // shader's operations in another order, which can move a last bit).
            ir::ResourceKind::StorageRead => {
                let ty = self.runtime_array(res.ty)?;
                format!("device {ty} const& {name}{}", at("buffer"))
            }
            ir::ResourceKind::StorageReadWrite => {
                let ty = self.runtime_array(res.ty)?;
                format!("device {ty}& {name}{}", at("buffer"))
            }
            ir::ResourceKind::Private => format!("thread {}& {name}", self.ty(res.ty)?),
            ir::ResourceKind::Workgroup => format!("threadgroup {}& {name}", self.ty(res.ty)?),
            ir::ResourceKind::Texture { depth, dim, format } => {
                let kind =
                    if ir::texel_scalar(format) == ir::Scalar::U32 { "uint" } else { "float" };
                let ty = match (depth, dim) {
                    (true, ir::TexDim::D2) => "depth2d".to_string(),
                    (true, ir::TexDim::D2Array) => "depth2d_array".to_string(),
                    (true, ir::TexDim::Cube) => "depthcube".to_string(),
                    (true, ir::TexDim::D3) => return Err("internal: a 3D depth texture".into()),
                    (false, ir::TexDim::D2) => "texture2d".to_string(),
                    (false, ir::TexDim::D2Array) => "texture2d_array".to_string(),
                    (false, ir::TexDim::Cube) => "texturecube".to_string(),
                    (false, ir::TexDim::D3) => "texture3d".to_string(),
                };
                let kind = if depth { "float" } else { kind };
                format!("metal::{ty}<{kind}, metal::access::sample> {name}{}", at("texture"))
            }
            ir::ResourceKind::StorageTexture { dim, format } => {
                let kind =
                    if ir::texel_scalar(format) == ir::Scalar::U32 { "uint" } else { "float" };
                let ty = match dim {
                    ir::TexDim::D2 => "texture2d",
                    ir::TexDim::D2Array => "texture2d_array",
                    ir::TexDim::D3 => "texture3d",
                    ir::TexDim::Cube => return Err("internal: a cube a kernel writes".into()),
                };
                format!("metal::{ty}<{kind}, metal::access::write> {name}{}", at("texture"))
            }
            ir::ResourceKind::Sampler { .. } => format!("metal::sampler {name}{}", at("sampler")),
        })
    }

    /// The one-member struct a texture or sampler of type `ty` is read from the table as.
    fn table_type(&mut self, ty: &str) -> String {
        if let Some(name) = self.table_types.get(ty) {
            return name.clone();
        }
        let name = format!("_T{}", self.table_types.len());
        let _ = writeln!(self.type_defs, "struct {name} {{\n{ty} v;\n}};");
        self.table_types.insert(ty.to_string(), name.clone());
        name
    }

    /// The pipeline's block of indices as a struct (`_Block`), written once: a `uint` for each
    /// binding the device's table holds, in [`wrela_abi::table::indexed`]'s order.
    fn block_struct(&mut self) {
        if self.block_written {
            return;
        }
        let mut members = String::new();
        for b in wrela_abi::table::indexed(self.p) {
            let _ = writeln!(members, "uint i{b};");
        }
        let _ = writeln!(self.type_defs, "struct _Block {{\n{members}}};");
        self.block_written = true;
    }

    /// The type of a sized buffer of `elem`s: an array of one, named once.
    fn runtime_array(&mut self, elem: ir::TypeId) -> R<String> {
        let name = format!("B{}", elem.0);
        if !self.runtime_arrays.contains(&elem) {
            let ty = self.ty(elem)?;
            let _ = writeln!(self.type_defs, "typedef {ty} {name}[1];");
            self.runtime_arrays.insert(elem);
        }
        Ok(name)
    }

    fn uniform_space(&self) -> Option<UniformSpace> {
        self.p.uniform.as_ref().map(|u| u.space)
    }

    /// Whether resource `r` is passed to the functions that use it (not a table).
    fn passed(&self, r: usize) -> bool {
        let _ = r;
        true
    }

    // ---- the module ----

    fn module(&mut self) -> R<()> {
        let m = self.m;
        let entries: HashSet<ir::FuncId> = m.entry_points.iter().map(|e| e.function).collect();
        // The functions that stay calls, callees first.
        let mut done = HashSet::new();
        let mut order = Vec::new();
        fn helpers_of(
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
            self.function(f, None)?;
        }
        for e in &m.entry_points {
            self.function(e.function, Some(e))?;
        }
        Ok(())
    }

    fn function(&mut self, id: ir::FuncId, entry: Option<&'m ir::EntryPoint>) -> R<()> {
        let f = &self.m.functions[id.index()];
        let mut fw = FunctionWriter::new(self, f, id, entry)?;
        let text = fw.write()?;
        self.functions.push_str(&text);
        Ok(())
    }

    fn stage_of(e: &ir::EntryPoint) -> ShaderStage {
        match e.stage {
            ir::Stage::Compute { .. } => ShaderStage::Compute,
            ir::Stage::Vertex { .. } => ShaderStage::Vertex,
            ir::Stage::Fragment { .. } => ShaderStage::Fragment,
        }
    }
}

/// A struct's name, cut to 32 characters (a creature's nested types' names run to hundreds).
fn short(name: &str) -> String {
    let mut name = ir::ident(name);
    if let Some((end, _)) = name.char_indices().nth(32) {
        name.truncate(end);
    }
    name.truncate(name.trim_end_matches('_').len());
    name
}

/// The resources an expression names outside its places.
fn expr_resources(e: &ir::Expr, used: &mut BTreeSet<usize>) {
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

/// A loop the MSL writer writes as `for` (`ir::counted`): its counter, limit and step, its test's
/// statement `check`, and the test's own statements nothing else reads (`unused`: a value nothing
/// reads is an error under `-Werror`).
struct Counted {
    local: ir::LocalId,
    check: usize,
    unused: Vec<usize>,
    limit: ir::Const,
    step: ir::Const,
}

/// [`Counted`] of a loop of `f`, if it is one.
fn counted(f: &ir::Function, body: &[ir::Stmt], continuing: &[ir::Stmt]) -> Option<Counted> {
    let c = ir::counted::counted(&ir::counted::constants(f), &[], body, continuing)?;
    // What the rest of the body reads; then each of the test's statements whose value that
    // reads stays, with what it reads.
    let test: std::collections::HashMap<usize, ir::ValueId> =
        c.test.iter().map(|&(v, at)| (at, v)).collect();
    let mut used = std::collections::HashSet::new();
    for (i, s) in body.iter().enumerate() {
        if i == c.check || test.contains_key(&i) {
            continue;
        }
        ir::visit::walk(std::slice::from_ref(s), &mut |x| {
            x.for_each_value(&mut |v| {
                used.insert(v);
            })
        });
    }
    let mut kept: Vec<usize> = Vec::new();
    loop {
        let more: Vec<usize> = test
            .iter()
            .filter(|(at, v)| !kept.contains(at) && used.contains(*v))
            .map(|(at, _)| *at)
            .collect();
        if more.is_empty() {
            break;
        }
        for at in more {
            body[at].for_each_value(&mut |v| {
                used.insert(v);
            });
            kept.push(at);
        }
    }
    let mut unused: Vec<usize> = test.keys().filter(|at| !kept.contains(at)).copied().collect();
    unused.sort_unstable();
    Some(Counted { local: c.local, check: c.check, unused, limit: c.limit, step: c.step })
}

/// A scalar constant as MSL writes it: a float in its shortest decimal that reads back the same
/// (with `.0` when whole), `INFINITY` and `NAN` by name; integers suffixed.
fn literal(c: &ir::Const) -> R<String> {
    Ok(match c {
        ir::Const::Bool(b) => b.to_string(),
        ir::Const::I32(x) if *x == i32::MIN => format!("({} - 1)", x + 1),
        ir::Const::I32(x) => x.to_string(),
        ir::Const::U32(x) => format!("{x}u"),
        ir::Const::F32(x) => float_literal(*x),
        ir::Const::F16(x) => {
            let h = half::f16::from_f32(*x);
            if h.is_infinite() {
                format!("{}INFINITY", if h.is_sign_negative() { "-" } else { "" })
            } else if h.is_nan() {
                "NAN".into()
            } else if h.to_f32().fract() == 0.0 {
                format!("{h}.0h")
            } else {
                format!("{h}h")
            }
        }
        other => return Err(format!("internal: the constant {other:?} in GPU code")),
    })
}

fn float_literal(x: f32) -> String {
    if x.is_infinite() {
        format!("{}INFINITY", if x.is_sign_negative() { "-" } else { "" })
    } else if x.is_nan() {
        "NAN".into()
    } else if x.fract() == 0.0 {
        format!("{x}.0")
    } else {
        format!("{x}")
    }
}

/// Writes helper `h`'s definition.
fn write_helper(out: &mut String, h: &Helper) {
    match h {
        Helper::Div(t, signed) => {
            let _ = writeln!(out, "{t} naga_div({t} lhs, {t} rhs) {{");
            if *signed {
                let _ = writeln!(
                    out,
                    "return lhs / metal::select(rhs, 1, (lhs == (-2147483647 - 1) & rhs == -1) | (rhs == 0));"
                );
            } else {
                let _ = writeln!(out, "return lhs / metal::select(rhs, 1u, rhs == 0u);");
            }
            out.push_str("}\n\n");
        }
        Helper::Mod(t, signed) => {
            let _ = writeln!(out, "{t} naga_mod({t} lhs, {t} rhs) {{");
            if *signed {
                let _ = writeln!(
                    out,
                    "{t} divisor = metal::select(rhs, 1, (lhs == (-2147483647 - 1) & rhs == -1) | (rhs == 0));"
                );
                out.push_str("return lhs - (lhs / divisor) * divisor;\n");
            } else {
                let _ = writeln!(out, "return lhs % metal::select(rhs, 1u, rhs == 0u);");
            }
            out.push_str("}\n\n");
        }
        Helper::Neg(t, u) => {
            let _ = writeln!(
                out,
                "{t} naga_neg({t} val) {{\nreturn as_type<{t}>(-as_type<{u}>(val));\n}}\n"
            );
        }
        Helper::Abs(t, u) => {
            let _ = writeln!(
                out,
                "{t} naga_abs({t} val) {{\nreturn metal::select(as_type<{t}>(-as_type<{u}>(val)), val, val >= 0);\n}}\n"
            );
        }
        Helper::F2I(from, to, signed) => {
            let (name, lo, hi) = if *signed {
                ("naga_f2i32", float_literal(-2147483648.0), float_literal(2147483520.0))
            } else {
                ("naga_f2u32", float_literal(0.0), float_literal(4294967040.0))
            };
            let _ = writeln!(
                out,
                "{to} {name}({from} value) {{\nreturn static_cast<{to}>(metal::clamp(value, {lo}, {hi}));\n}}\n"
            );
        }
        Helper::Dot(vec, scalar, n) => {
            let name = format!("naga_dot_{}{n}", scalar);
            let _ = write!(out, "{scalar} {name}({vec} a, {vec} b) {{\nreturn (");
            for k in 0..usize::from(*n) {
                let c = ["x", "y", "z", "w"][k];
                let _ = write!(out, " + a.{c} * b.{c}");
            }
            out.push_str(");\n}\n\n");
        }
        Helper::CompareExchange(scalar, result) => {
            for space in ["device", "threadgroup"] {
                let _ = writeln!(
                    out,
                    "template <typename A>\n{result} naga_atomic_compare_exchange_weak_explicit({space} A *atomic_ptr, {scalar} cmp, {scalar} v) {{\n\
                     bool swapped = metal::atomic_compare_exchange_weak_explicit(atomic_ptr, &cmp, v, metal::memory_order_relaxed, metal::memory_order_relaxed);\n\
                     return {result}{{cmp, swapped}};\n}}"
                );
            }
        }
    }
}

// ---- functions ---------------------------------------------------------------------------------------

/// How a value is written where it's used: its name, or its expression.
#[derive(Clone)]
struct Val {
    text: String,
    /// The text is a name or otherwise binds tightest: it needs no parentheses as an operand.
    atom: bool,
}

/// Writes one function: a helper that stays a call, or an entry point.
struct FunctionWriter<'a, 'm> {
    w: &'a mut Writer<'m>,
    f: &'m ir::Function,
    id: ir::FuncId,
    entry: Option<&'m ir::EntryPoint>,
    out: String,
    vals: Vec<Option<Val>>,
    /// How many times each value is used.
    uses: Vec<u32>,
    locals: Vec<String>,
    params: Vec<String>,
    /// Each builtin input's argument name (`None`: a fragment's position, which is its own).
    inputs: Vec<String>,
    /// A vertex shader's output struct, or a fragment shader's input struct: its name, and the
    /// members (`v{k}`) of each varying, with the clip position's index.
    varyings: Option<(&'m [ir::Varying], ir::TypeId)>,
    /// A fragment shader's output struct's name.
    frag_out: Option<String>,
    vertex_out: Option<String>,
    /// The fragment's position argument, when it takes varyings.
    position_arg: Option<String>,
    /// Names made inside the function (loop bounds, levels): counted.
    fresh: u32,
    /// Whether the function reads a sized buffer's length (it takes the sizes buffer).
    takes_sizes: bool,
    /// The integer constants among its values: an index known to be in range isn't clamped.
    consts: HashMap<ir::ValueId, i64>,
    /// How many loops the statement being written is in.
    loop_depth: u32,
}

/// Whether `e` is written where it's used: it only computes from values, and isn't a derivative
/// (which wants the control flow it's in) or an element at a dynamic index.
fn pure(e: &ir::Expr) -> bool {
    e.only_computes()
        && !matches!(
            e,
            ir::Expr::ExtractDyn(..)
                | ir::Expr::Builtin(ir::Builtin::Dpdx | ir::Builtin::Dpdy | ir::Builtin::Fwidth, _)
        )
}

/// Whether `e` is written where it's used however often it's used, never bound to a name: a
/// constant, or a part of a value (as naga wrote them, which binds neither: under Metal's fast
/// math the order the compiler meets operations in can move a sum's last bit).
fn written_at_each_use(e: &ir::Expr) -> bool {
    matches!(e, ir::Expr::Const(_) | ir::Expr::Zero(_) | ir::Expr::Extract(..))
}

impl<'a, 'm> FunctionWriter<'a, 'm> {
    fn new(
        w: &'a mut Writer<'m>,
        f: &'m ir::Function,
        id: ir::FuncId,
        entry: Option<&'m ir::EntryPoint>,
    ) -> R<FunctionWriter<'a, 'm>> {
        let mut uses = vec![0u32; f.values.len()];
        ir::visit::walk(&f.body, &mut |s| {
            s.for_each_value(&mut |v| uses[v.index()] += 1);
        });
        let locals = f.locals.iter().enumerate().map(|(k, l)| named("l", &l.name, k)).collect();
        let params = f.params.iter().enumerate().map(|(k, p)| named("p", &p.name, k)).collect();
        let takes_sizes = w.uses[id.index()].iter().any(|&r| is_sized(&w.m.resources[r]));
        Ok(FunctionWriter {
            w,
            f,
            id,
            entry,
            out: String::new(),
            vals: vec![None; f.values.len()],
            uses,
            locals,
            params,
            inputs: Vec::new(),
            varyings: None,
            frag_out: None,
            vertex_out: None,
            position_arg: None,
            fresh: 0,
            takes_sizes,
            consts: HashMap::new(),
            loop_depth: 0,
        })
    }

    fn fresh(&mut self, stem: &str) -> String {
        self.fresh += 1;
        format!("{stem}{}", self.fresh)
    }

    fn write(&mut self) -> R<String> {
        let mut header = String::new();
        match self.entry {
            None => self.helper_signature(&mut header)?,
            Some(e) => self.entry_signature(e, &mut header)?,
        }
        // (The IR's own body, not a copy: what the compiler proves of its loops is by address.)
        let body = &self.f.body;
        let mut text = String::new();
        // Locals, then the body.
        for (k, l) in self.f.locals.iter().enumerate() {
            let ty = self.w.ty(l.ty)?;
            let _ = writeln!(text, "{ty} {} = {{}};", self.locals[k]);
        }
        self.block(body, &mut text)?;
        if let Some(ir::Stmt::Trap) = body.last() {
            self.return_zero(&mut text)?;
        }
        let mut out = header;
        out.push_str(&self.out);
        out.push_str(&text);
        out.push_str("}\n\n");
        Ok(out)
    }

    /// A function that stays a call: `T name(params, resources it uses, sizes) {`.
    fn helper_signature(&mut self, out: &mut String) -> R<()> {
        let ret = match self.f.ret {
            Some(r) => self.w.ty(r)?,
            None => "void".into(),
        };
        let mut args = Vec::new();
        for (k, p) in self.f.params.iter().enumerate() {
            if p.by_ref {
                return Err("internal: a place passed to a called GPU function".into());
            }
            args.push(format!("{} {}", self.w.ty(p.ty)?, self.params[k]));
        }
        for r in self.w.uses[self.id.index()].clone() {
            if self.w.passed(r) {
                args.push(self.w.resource_arg(r, None)?);
            }
        }
        if self.takes_sizes {
            args.push("constant _mslBufferSizes& _buffer_sizes".into());
        }
        let name = &self.w.function_names[self.id.index()];
        let _ = writeln!(out, "{ret} {name}(\n{}\n) {{", args.join("\n, "));
        Ok(())
    }

    /// An entry point's signature, its private and workgroup variables, and what rebuilds its
    /// inputs.
    fn entry_signature(&mut self, e: &'m ir::EntryPoint, out: &mut String) -> R<()> {
        let name = entry_name(&e.name);
        self.w.entry_names.push(name.clone());
        let stage = Writer::stage_of(e);
        let layout = slots::layout(self.w.p);
        let st = slots::slots(&layout, stage);
        let mut slots_max = wrela_abi::TargetLimits::METAL.buffer_slots_per_stage.unwrap_or(31);
        if !wrela_abi::table::indexed(self.w.p).is_empty() {
            // The block of indices and the table take the last two.
            slots_max -= 2;
        }
        if st.buffers() > slots_max {
            return Err(format!(
                "internal: `{}` binds {} buffers, past Metal's {}",
                e.name,
                st.buffers(),
                slots_max
            ));
        }
        let mut args = Vec::new();
        let mut prelude = String::new();
        // Builtin inputs first, in order.
        let in_varyings = |b: &ir::BuiltinInput| {
            *b == ir::BuiltinInput::Position
                && matches!(e.stage, ir::Stage::Fragment { varyings: Some(_), .. })
        };
        for (i, b) in e.inputs.iter().enumerate() {
            use ir::BuiltinInput as B;
            let arg = format!("input{i}");
            self.inputs.push(arg.clone());
            if in_varyings(b) {
                continue;
            }
            let (ty, attr) = match b {
                B::GlobalInvocationId => ("metal::uint3", "thread_position_in_grid"),
                B::LocalInvocationId => ("metal::uint3", "thread_position_in_threadgroup"),
                B::WorkgroupId => ("metal::uint3", "threadgroup_position_in_grid"),
                B::NumWorkgroups => ("metal::uint3", "threadgroups_per_grid"),
                B::VertexIndex => ("uint", "vertex_id"),
                B::InstanceIndex => ("uint", "instance_id"),
                B::Position => ("metal::float4", "position"),
                B::SubgroupInvocationId => ("uint", "thread_index_in_simdgroup"),
                B::SubgroupSize => ("uint", "threads_per_simdgroup"),
            };
            args.push(format!("{ty} {arg} [[{attr}]]"));
        }
        let mut result = "void".to_string();
        match &e.stage {
            ir::Stage::Compute { .. } => {}
            ir::Stage::Vertex { varyings } => {
                let rt = self.f.ret.ok_or("internal: a vertex shader with no output")?;
                let out_name = format!("{name}Output");
                let mut def = format!("struct {out_name} {{\n");
                let mut location = 0;
                for (k, v) in varyings.iter().enumerate() {
                    let ty = self.w.ty(part_ty(&self.w.m.types, rt, &v.path)?)?;
                    if v.position {
                        let _ = writeln!(def, "{ty} v{k} [[position]];");
                    } else {
                        let interp = if v.flat { "flat" } else { "center_perspective" };
                        let _ = writeln!(def, "{ty} v{k} [[user(loc{location}), {interp}]];");
                        location += 1;
                    }
                }
                def.push_str("};\n");
                self.w.type_defs.push_str(&def);
                self.varyings = Some((varyings, rt));
                self.vertex_out = Some(out_name.clone());
                result = out_name;
            }
            ir::Stage::Fragment { varyings, outputs, depth, .. } => {
                if let Some((vt, map)) = varyings {
                    let in_name = format!("{name}Input");
                    let mut def = format!("struct {in_name} {{\n");
                    let mut location = 0;
                    for (k, v) in map.iter().enumerate() {
                        if v.position {
                            continue;
                        }
                        let ty = self.w.ty(part_ty(&self.w.m.types, *vt, &v.path)?)?;
                        let interp = if v.flat { "flat" } else { "center_perspective" };
                        let _ = writeln!(def, "{ty} v{k} [[user(loc{location}), {interp}]];");
                        location += 1;
                    }
                    def.push_str("};\n");
                    self.w.type_defs.push_str(&def);
                    args.insert(0, format!("{in_name} varyings [[stage_in]]"));
                    let position = "position".to_string();
                    args.insert(1, format!("metal::float4 {position} [[position]]"));
                    self.position_arg = Some(position);
                    self.varyings = Some((map, *vt));
                }
                let out_name = format!("{name}Output");
                let mut def = format!("struct {out_name} {{\n");
                for (location, c) in outputs.iter().enumerate() {
                    let ty = vector_name(c.scalar, 4)?;
                    let _ = writeln!(def, "{ty} color{location} [[color({location})]];");
                }
                if depth.is_some() {
                    let q = self.w.depth.map_or("depth(any)", ConservativeDepth::qualifier);
                    let _ = writeln!(def, "float depth [[{q}]];");
                }
                def.push_str("};\n");
                self.w.type_defs.push_str(&def);
                self.frag_out = Some(out_name.clone());
                result = out_name;
            }
        }
        // The resources it uses, each in its slot, or made from the device's table.
        let used = self.w.uses[self.id.index()].clone();
        let mut workgroup = Vec::new();
        let mut locals = String::new();
        for &r in &used {
            let res = &self.w.m.resources[r];
            match res.kind {
                ir::ResourceKind::Private => {
                    let ty = self.w.ty(res.ty)?;
                    let _ = writeln!(prelude, "{ty} {} = {{}};", self.w.resource_names[r]);
                }
                ir::ResourceKind::Workgroup => workgroup.push(r),
                _ if wrela_abi::table::index_offset(self.w.p, res.binding).is_some() => {
                    let binding = res.binding;
                    let arg = self.w.resource_arg(r, None)?;
                    let name = &self.w.resource_names[r];
                    let ty = arg
                        .strip_suffix(name.as_str())
                        .ok_or("internal: a resource's type")?
                        .trim_end()
                        .to_string();
                    let one = self.w.table_type(&ty);
                    let _ = writeln!(
                        locals,
                        "{arg} = reinterpret_cast<constant {one}*>(_table)[_blk.i{binding}].v;"
                    );
                }
                _ => {
                    let slot = st.of(res.binding).copied().ok_or_else(|| {
                        format!(
                            "internal: `{}` uses binding {}, which its stage doesn't see",
                            e.name, res.binding
                        )
                    })?;
                    let expected = match slot.table {
                        Table::Buffer => matches!(
                            res.kind,
                            ir::ResourceKind::Uniform { .. }
                                | ir::ResourceKind::StorageRead
                                | ir::ResourceKind::StorageReadWrite
                        ),
                        Table::Texture => matches!(
                            res.kind,
                            ir::ResourceKind::Texture { .. }
                                | ir::ResourceKind::StorageTexture { .. }
                        ),
                        Table::Sampler => matches!(res.kind, ir::ResourceKind::Sampler { .. }),
                    };
                    if !expected {
                        return Err(format!(
                            "internal: binding {} of `{}` is in the wrong argument table",
                            res.binding, e.name
                        ));
                    }
                    args.push(self.w.resource_arg(r, Some(&slot))?);
                }
            }
        }
        if !locals.is_empty() {
            self.w.block_struct();
            args.push(format!(
                "constant _Block& _blk [[buffer({})]]",
                wrela_abi::table::METAL_BLOCK
            ));
            args.push(format!(
                "constant ulong* _table [[buffer({})]]",
                wrela_abi::table::METAL_TABLE
            ));
            prelude.push_str(&locals);
        }
        if self.takes_sizes {
            let sizes =
                self.w.p.sizes.as_ref().ok_or("internal: a pipeline with no sizes binding")?;
            let slot = st.of(sizes.binding).ok_or("internal: the sizes binding has no slot")?;
            args.push(format!(
                "constant _mslBufferSizes& _buffer_sizes [[buffer({})]]",
                slot.index
            ));
        }
        // Workgroup memory: declared in the kernel, zeroed by its first invocation before any
        // reads it (WGSL's rule, as naga wrote it).
        if !workgroup.is_empty() {
            args.push("uint __local_invocation_index [[thread_index_in_threadgroup]]".into());
            let mut decl = String::new();
            let mut zero = String::new();
            for &r in &workgroup {
                let res = &self.w.m.resources[r];
                let ty = self.w.ty(res.ty)?;
                let name = self.w.resource_names[r].clone();
                let _ = writeln!(decl, "threadgroup {ty} {name};");
                self.zero_workgroup(res.ty, name, &mut zero, 0)?;
            }
            let _ = writeln!(
                prelude,
                "{decl}if (__local_invocation_index == 0u) {{\n{zero}}}\nmetal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);"
            );
        }
        let qualifier = match e.stage {
            ir::Stage::Compute { workgroup_size } => {
                let total = workgroup_size[0] * workgroup_size[1] * workgroup_size[2];
                format!("[[max_total_threads_per_threadgroup({total})]] kernel")
            }
            ir::Stage::Vertex { .. } => "vertex".into(),
            ir::Stage::Fragment { .. } => "fragment".into(),
        };
        let _ = writeln!(out, "{qualifier} {result} {name}(\n{}\n) {{", args.join("\n, "));
        out.push_str(&prelude);
        Ok(())
    }

    /// Zeroes workgroup variable `place` of type `t`: what a brace can't (an atomic, or an array
    /// or struct holding one) member by member.
    fn zero_workgroup(
        &mut self,
        t: ir::TypeId,
        place: String,
        out: &mut String,
        depth: u32,
    ) -> R<()> {
        let types = &self.w.m.types;
        if !holds_atomic(types, t) {
            let _ = writeln!(out, "{place} = {{}};");
            return Ok(());
        }
        match types.get(t).clone() {
            ir::TypeDef::Atomic(_) => {
                let _ = writeln!(
                    out,
                    "metal::atomic_store_explicit(&{place}, 0, metal::memory_order_relaxed);"
                );
            }
            ir::TypeDef::Array(e, n) => {
                let _ =
                    writeln!(out, "for (int __i{depth} = 0; __i{depth} < {n}; __i{depth}++) {{");
                self.zero_workgroup(e, format!("{place}.inner[__i{depth}]"), out, depth + 1)?;
                out.push_str("}\n");
            }
            ir::TypeDef::Struct { .. } | ir::TypeDef::Enum { .. } => {
                let members = self.w.members(t)?;
                for (k, (_, mt)) in members.iter().enumerate() {
                    let (name, _) = self.w.member_name(t, k as u32)?;
                    self.zero_workgroup(*mt, format!("{place}.{name}"), out, depth)?;
                }
            }
            _ => return Err("internal: an atomic in an unexpected place".into()),
        }
        Ok(())
    }

    // ---- values ----

    fn val(&self, v: ir::ValueId) -> R<Val> {
        self.vals
            .get(v.index())
            .cloned()
            .flatten()
            .ok_or_else(|| format!("internal: v{} has no MSL expression", v.0))
    }

    /// `v` as an operand: parenthesized unless it binds tightest.
    fn operand(&self, v: ir::ValueId) -> R<String> {
        let val = self.val(v)?;
        Ok(if val.atom { val.text } else { format!("({})", val.text) })
    }

    /// `v` as a whole argument (a call's, an initializer's): as it is.
    fn arg(&self, v: ir::ValueId) -> R<String> {
        Ok(self.val(v)?.text)
    }

    /// The loop body `b`'s leading exit test, where it has one a `for`'s condition can hold:
    /// pure `let`s (none reading the counter `counter`, set only after the condition), each read
    /// only by them and the test, then `if cond { break }`. Its condition as MSL, with the
    /// index of the test's `if` in `b`.
    fn leading_test(&mut self, b: &[ir::Stmt], counter: ir::LocalId) -> R<Option<(String, usize)>> {
        let mut lets: Vec<(ir::ValueId, &ir::Expr)> = Vec::new();
        for (i, s) in b.iter().enumerate() {
            match s {
                ir::Stmt::At(_) => {}
                ir::Stmt::Let(v, e) => {
                    let reads_counter =
                        matches!(e, ir::Expr::Load(p) if p.root_local() == Some(counter));
                    let simple =
                        matches!(e, ir::Expr::Load(p) if p.root_local().is_some()) || pure(e);
                    if reads_counter || !simple {
                        return Ok(None);
                    }
                    lets.push((*v, e));
                }
                ir::Stmt::If { cond, then, else_ }
                    if then.iter().filter(|s| !matches!(s, ir::Stmt::At(_))).count() == 1
                        && then.iter().any(|s| matches!(s, ir::Stmt::Break))
                        && else_.iter().all(|s| matches!(s, ir::Stmt::At(_))) =>
                {
                    // Every use of the lets' values is in the lets or the condition.
                    let mut inside: HashMap<ir::ValueId, u32> = HashMap::new();
                    for (_, e) in &lets {
                        e.for_each_value(&mut |x| *inside.entry(x).or_default() += 1);
                    }
                    *inside.entry(*cond).or_default() += 1;
                    if lets
                        .iter()
                        .any(|(v, _)| inside.get(v).copied().unwrap_or(0) != self.uses[v.index()])
                        || !lets.iter().any(|(v, _)| v == cond)
                    {
                        return Ok(None);
                    }
                    // Written where used: the lets add no statements.
                    let mut scratch = String::new();
                    for (v, e) in &lets {
                        let text = self.value(*v, e, &mut scratch)?;
                        self.vals[v.index()] = Some(text);
                    }
                    if !scratch.is_empty() {
                        return Ok(None);
                    }
                    return Ok(Some((self.arg(*cond)?, i)));
                }
                _ => return Ok(None),
            }
        }
        Ok(None)
    }

    fn ty_of(&self, v: ir::ValueId) -> ir::TypeId {
        self.f.value_ty(v)
    }

    fn scalar_of(&self, v: ir::ValueId) -> Option<ir::Scalar> {
        self.w.m.types.element_scalar(self.ty_of(v))
    }

    /// The MSL type name of `v`'s type, scalar or vector, with its scalar replaced by `s`.
    fn numeric_like(&self, t: ir::TypeId, s: ir::Scalar) -> R<String> {
        match self.w.m.types.get(t) {
            ir::TypeDef::Vector(_, n) => vector_name(s, *n),
            _ => Ok(scalar_name(s)?.to_string()),
        }
    }

    // ---- statements ----

    fn block(&mut self, b: &[ir::Stmt], out: &mut String) -> R<()> {
        for s in b {
            self.stmt(s, out)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &ir::Stmt, out: &mut String) -> R<()> {
        match s {
            ir::Stmt::Let(v, e) => {
                let t = self.f.value_ty(*v);
                let text = self.value(*v, e, out)?;
                if self.uses[v.index()] == 0 {
                    // Nothing reads it: written only for what it does.
                    // (A compare-exchange's call is already written, to its own name.)
                    let exchanged =
                        matches!(e, ir::Expr::Atomic(ir::AtomicOp::CompareExchange, ..));
                    if e.has_effect() && !exchanged {
                        let _ = writeln!(out, "{};", text.text);
                    }
                } else if pure(e) && (self.uses[v.index()] <= 1 || written_at_each_use(e)) {
                    self.vals[v.index()] = Some(text);
                } else {
                    let name = format!("v{}", v.0);
                    let ty = self.w.ty(t)?;
                    let _ = writeln!(out, "{ty} {name} = {};", text.text);
                    self.vals[v.index()] = Some(Val { text: name, atom: true });
                }
            }
            ir::Stmt::Eval(e) => match e {
                ir::Expr::Call(g, args) => {
                    let call = self.call(*g, args)?;
                    let _ = writeln!(out, "{call};");
                }
                ir::Expr::Host(..) => return Err("GPU code can't record GPU work".into()),
                ir::Expr::Mem(..) => return Err("GPU code can't use raw memory".into()),
                ir::Expr::Barrier => {
                    out.push_str(
                        "metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);\n",
                    );
                }
                ir::Expr::Discard => out.push_str("metal::discard_fragment();\n"),
                ir::Expr::Atomic(op, p, args) => {
                    if let Some(text) = self.atomic(*op, p, args, None, out)? {
                        let _ = writeln!(out, "{text};");
                    }
                }
                ir::Expr::TextureStore(tex, args) => self.texture_store(*tex, args, out)?,
                _ => {}
            },
            ir::Stmt::Store(p, v) => {
                let (place, atomic) = self.place(p, out)?;
                let value = self.arg(*v)?;
                if atomic {
                    let _ = writeln!(
                        out,
                        "metal::atomic_store_explicit(&{place}, {value}, metal::memory_order_relaxed);"
                    );
                } else {
                    let _ = writeln!(out, "{place} = {value};");
                }
            }
            ir::Stmt::If { cond, then, else_ } => {
                let c = self.arg(*cond)?;
                let _ = writeln!(out, "if ({c}) {{");
                self.block(then, out)?;
                if !else_.is_empty() {
                    out.push_str("} else {\n");
                    self.block(else_, out)?;
                }
                out.push_str("}\n");
            }
            ir::Stmt::Loop { body, continuing } => {
                let bound = (self.w.checks.loops && !self.w.proofs.ends(body))
                    .then(|| self.fresh("loop_bound"));
                let gate = (!continuing.is_empty()).then(|| self.fresh("loop_init"));
                // A counted loop (`ir::counted`) that repeats (in a loop, or in a function some
                // loop calls: a march's every step) is written as a `for` Metal doesn't unroll:
                // its compiler does more with that shape than with a `while` whose `continuing`
                // runs at its top behind a flag (the masked march: 2.52 -> 2.13 ms), and unrolled
                // a large body is copied pass by pass (the engine's edge rays: 1.54 -> 7.74 ms).
                // One that runs once an invocation stays a `while`, as Metal does best with
                // there: as a `for` not unrolled, a ray's `meets` over a fold's 17 parts took
                // the engine's picture 7% longer; as one it may unroll, the kernels' loops took
                // its frame twice as long. backend.md, items 12 and 18.
                let repeats = self.loop_depth > 0 || self.w.repeated[self.id.index()];
                if bound.is_none()
                    && repeats
                    && let Some(c) = counted(self.f, body, continuing)
                {
                    let it = self.fresh("it");
                    let l = self.locals[c.local.index()].clone();
                    let ty = self.w.ty(self.f.locals[c.local.index()].ty)?;
                    let (limit, step) = (literal(&c.limit)?, literal(&c.step)?);
                    let _ = writeln!(out, "{ty} {it} = {l};");
                    let _ = writeln!(out, "#pragma clang loop unroll(disable)");
                    // The check goes, and the counter's read and the comparison where only the
                    // check used them (a value nothing reads is an error under `-Werror`).
                    let dropped = |i: usize| i == c.check || c.unused.contains(&i);
                    let mut rest: Vec<ir::Stmt> = body
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !dropped(*i))
                        .map(|(_, s)| s.clone())
                        .collect();
                    // A second test right after, of pure reads nothing else uses (the set-bit
                    // walk's `if bits == 0 { break }`), joins the condition: Metal's compiler
                    // does more with one exit (the masked march: 0.062 of its 0.089 ms).
                    let also = match self.leading_test(&rest, c.local)? {
                        Some((cond, upto)) => {
                            rest.drain(..=upto);
                            format!(" && !({cond})")
                        }
                        None => String::new(),
                    };
                    let _ = writeln!(out, "for (; {it} < {limit}{also}; {it} = {it} + {step}) {{");
                    let _ = writeln!(out, "{l} = {it};");
                    self.loop_depth += 1;
                    self.block(&rest, out)?;
                    self.loop_depth -= 1;
                    out.push_str("}\n");
                    let _ = writeln!(out, "{l} = {it};");
                    return Ok(());
                }
                if let Some(b) = &bound {
                    let _ = writeln!(out, "uint2 {b} = uint2({}u);", u32::MAX);
                }
                if let Some(g) = &gate {
                    let _ = writeln!(out, "bool {g} = true;");
                }
                out.push_str("while(true) {\n");
                if let Some(b) = &bound {
                    let _ = writeln!(
                        out,
                        "if (metal::all({b} == uint2(0u))) {{ break; }}\n{b} -= uint2({b}.y == 0u, 1u);"
                    );
                }
                self.loop_depth += 1;
                if let Some(g) = &gate {
                    let _ = writeln!(out, "if (!{g}) {{");
                    self.block(continuing, out)?;
                    out.push_str("}\n");
                    let _ = writeln!(out, "{g} = false;");
                }
                self.block(body, out)?;
                self.loop_depth -= 1;
                out.push_str("}\n");
            }
            ir::Stmt::Break => out.push_str("break;\n"),
            ir::Stmt::Continue => out.push_str("continue;\n"),
            ir::Stmt::Return(v) => match v {
                Some(v) => self.return_value(*v, out)?,
                None => out.push_str("return;\n"),
            },
            ir::Stmt::At(_) => {}
            // GPU code can't trap (its integers wrap, and its indexing is robust): a trap is a
            // point valid code never reaches.
            ir::Stmt::Trap => {}
        }
        Ok(())
    }

    /// A call to a function that stays one, with the resources it uses and the sizes.
    fn call(&mut self, g: ir::FuncId, args: &[ir::Arg]) -> R<String> {
        let mut texts = Vec::new();
        for a in args {
            match a {
                ir::Arg::Value(v) => texts.push(self.arg(*v)?),
                ir::Arg::Place(_) => {
                    return Err("internal: a place passed to a called GPU function".into());
                }
            }
        }
        for r in self.w.uses[g.index()].clone() {
            if self.w.passed(r) {
                texts.push(self.w.resource_names[r].clone());
            }
        }
        if self.w.uses[g.index()].iter().any(|&r| is_sized(&self.w.m.resources[r])) {
            texts.push("_buffer_sizes".into());
        }
        Ok(format!("{}({})", self.w.function_names[g.index()], texts.join(", ")))
    }

    fn return_zero(&mut self, out: &mut String) -> R<()> {
        if let Some(name) = self.frag_out.clone().or(self.vertex_out.clone()) {
            let _ = writeln!(out, "return {name} {{}};");
        } else if let Some(r) = self.f.ret {
            let _ = writeln!(out, "return {} {{}};", self.w.ty(r)?);
        } else {
            out.push_str("return;\n");
        }
        Ok(())
    }

    /// A return: a vertex shader's output struct from its value's parts, a fragment shader's
    /// colours (each four of its scalar) and depth, or the value.
    fn return_value(&mut self, v: ir::ValueId, out: &mut String) -> R<()> {
        let value = self.arg(v)?;
        if let Some(name) = self.frag_out.clone() {
            let Some(ir::EntryPoint { stage: ir::Stage::Fragment { outputs, depth, .. }, .. }) =
                self.entry
            else {
                return Err("internal: a fragment output outside a fragment shader".into());
            };
            let tmp = self.fresh("_tmp");
            let ty = self.w.ty(self.ty_of(v))?;
            let _ = writeln!(out, "const {ty} {tmp} = {value};");
            let mut parts = Vec::new();
            for c in outputs {
                let (part, _) = self.part_text(&tmp, self.ty_of(v), &c.path)?;
                parts.push(widened(&part, c)?);
            }
            if let Some(path) = depth {
                parts.push(self.part_text(&tmp, self.ty_of(v), path)?.0);
            }
            let _ = writeln!(out, "return {name} {{ {} }};", parts.join(", "));
            return Ok(());
        }
        if let (Some(name), Some((map, _))) = (self.vertex_out.clone(), self.varyings) {
            let tmp = self.fresh("_tmp");
            let ty = self.w.ty(self.ty_of(v))?;
            let _ = writeln!(out, "const {ty} {tmp} = {value};");
            let mut parts = Vec::new();
            for var in map {
                parts.push(self.part_text(&tmp, self.ty_of(v), &var.path)?.0);
            }
            let _ = writeln!(out, "return {name} {{ {} }};", parts.join(", "));
            return Ok(());
        }
        let _ = writeln!(out, "return {value};");
        Ok(())
    }

    /// The text of the part of `base` (of type `t`) at `path` (field indexes, then a matrix
    /// column's), a packed vector converted; and its type.
    fn part_text(&mut self, base: &str, t: ir::TypeId, path: &[u32]) -> R<(String, ir::TypeId)> {
        let mut text = base.to_string();
        let mut t = t;
        for &k in path {
            let types = &self.w.m.types;
            match types.get(t).clone() {
                ir::TypeDef::Struct { .. } | ir::TypeDef::Enum { .. } => {
                    let member = self.w.member(t, k);
                    let (name, packed) = self.w.member_name(t, member)?;
                    let next = self.w.m.types.part(t, k).ok_or("internal: a varying's path")?;
                    text = format!("{text}.{name}");
                    if packed {
                        let ty = self.w.ty(next)?;
                        text = format!("{ty}({text})");
                    }
                    t = next;
                }
                ir::TypeDef::Matrix(..) => {
                    text = format!("{text}[{k}]");
                    t = self.w.m.types.part(t, k).ok_or("internal: a varying's path")?;
                }
                _ => return Err("internal: a varying's path".into()),
            }
        }
        Ok((text, t))
    }

    // ---- places ----

    /// An index into something of `len` elements (`None`: a sized buffer's, read from the sizes
    /// buffer), clamped into range unless it's a constant known to be inside.
    fn index_text(
        &mut self,
        i: ir::ValueId,
        len: Option<u32>,
        sized: Option<(usize, ir::TypeId)>,
    ) -> R<String> {
        let text = self.arg(i)?;
        if !self.w.checks.bounds {
            return Ok(text);
        }
        if let (Some(n), Some(k)) = (len, self.const_int(i)) {
            if (0..i64::from(n)).contains(&k) {
                return Ok(text);
            }
        }
        if let Some(n) = len
            && self.w.proofs.in_range(self.f, i, n)
        {
            return Ok(text);
        }
        let limit = match (len, sized) {
            (Some(n), _) => format!("{}u", n - 1),
            (None, Some((r, elem))) => self.max_index(r, elem)?,
            (None, None) => return Err("internal: an index with no length".into()),
        };
        Ok(format!("metal::min(unsigned({text}), {limit})"))
    }

    /// The largest index into sized buffer `r` of elements of type `elem`: its bound length,
    /// less one element, over the stride.
    fn max_index(&mut self, r: usize, elem: ir::TypeId) -> R<String> {
        let member =
            self.w.size_members.get(&r).cloned().ok_or("internal: a buffer with no size")?;
        let size = msl_size(&self.w.m.types, &mut self.w.layouts, elem);
        let size = match self.w.m.types.get(elem) {
            // naga sized a vec3 element 12 bytes.
            ir::TypeDef::Vector(s, 3) => s.bytes() * 3,
            _ => size,
        };
        let stride = self.w.layouts.get(&self.w.m.types, elem).0.stride();
        Ok(format!("(_buffer_sizes.{member} - 0 - {size}) / {stride}"))
    }

    /// The integer constant `v` is, if it's one.
    fn const_int(&self, v: ir::ValueId) -> Option<i64> {
        self.consts.get(&v).copied()
    }

    /// A place's text, and whether it's an atomic.
    fn place(&mut self, p: &ir::Place, out: &mut String) -> R<(String, bool)> {
        let _ = out;
        let m = self.w.m;
        let (mut text, mut t, mut sized) = match &p.root {
            ir::PlaceRoot::Local(l) => {
                (self.locals[l.index()].clone(), Some(self.f.locals[l.index()].ty), None)
            }
            ir::PlaceRoot::Param(i) => {
                (self.params[*i as usize].clone(), Some(self.f.params[*i as usize].ty), None)
            }
            ir::PlaceRoot::Resource(r) => {
                let res = &m.resources[r.index()];
                let name = self.w.resource_names[r.index()].clone();
                if is_sized(res) {
                    (name, None, Some((r.index(), res.ty)))
                } else {
                    (name, Some(res.ty), None)
                }
            }
            ir::PlaceRoot::Data(d) => {
                let name = self.w.data(*d)?;
                (name, Some(m.data[d.index()].ty), None)
            }
            ir::PlaceRoot::Ptr(_) => {
                return Err(
                    "projections returned from functions aren't supported on the GPU".into()
                );
            }
        };
        for proj in &p.path {
            match (proj, t) {
                // A sized buffer's element.
                (ir::Proj::Index(i), None) => {
                    let (r, elem) = sized.take().ok_or("internal: a place's type")?;
                    let index = self.index_text(*i, None, Some((r, elem)))?;
                    text = format!("{text}[{index}]");
                    t = Some(elem);
                }
                (ir::Proj::Field(k), Some(pt)) => {
                    let member = self.w.member(pt, *k);
                    let (name, _) = self.w.member_name(pt, member)?;
                    text = format!("{text}.{name}");
                    t = m.proj_ty(pt, proj);
                }
                (ir::Proj::Comp(c), Some(pt)) => {
                    text = format!("{text}.{}", ["x", "y", "z", "w"][*c as usize]);
                    t = m.proj_ty(pt, proj);
                }
                (ir::Proj::Index(i), Some(pt)) => {
                    let len = static_len(&m.types, pt);
                    let index = self.index_text(*i, len, None)?;
                    let inner = if matches!(m.types.get(pt), ir::TypeDef::Array(..)) {
                        ".inner"
                    } else {
                        ""
                    };
                    text = format!("{text}{inner}[{index}]");
                    t = m.proj_ty(pt, proj);
                }
                _ => return Err("internal: a place's type".into()),
            }
        }
        let atomic = t.is_some_and(|t| matches!(m.types.get(t), ir::TypeDef::Atomic(_)));
        Ok((text, atomic))
    }
}

fn holds_atomic(types: &ir::Types, t: ir::TypeId) -> bool {
    match types.get(t) {
        ir::TypeDef::Atomic(_) => true,
        ir::TypeDef::Array(e, _) => holds_atomic(types, *e),
        ir::TypeDef::Struct { fields, .. } => fields.iter().any(|(_, f)| holds_atomic(types, *f)),
        ir::TypeDef::Enum { variants, .. } => {
            variants.iter().any(|(_, p)| p.is_some_and(|p| holds_atomic(types, p)))
        }
        _ => false,
    }
}

/// How many elements an index into a value of type `t` can reach, when the type says.
fn static_len(types: &ir::Types, t: ir::TypeId) -> Option<u32> {
    match types.get(t) {
        ir::TypeDef::Array(_, n) => Some(*n),
        ir::TypeDef::Vector(_, n) | ir::TypeDef::Matrix(n, _) => Some(u32::from(*n)),
        _ => None,
    }
}

/// The type of the part of a value of type `t` at `path`: field indexes, then a matrix column's.
fn part_ty(types: &ir::Types, t: ir::TypeId, path: &[u32]) -> R<ir::TypeId> {
    path.iter()
        .try_fold(t, |t, &k| types.part(t, k))
        .ok_or_else(|| "internal: a varying's path".to_string())
}

/// A colour of `c.components` of `c.scalar` as four of it: missing colour channels 0, and a
/// float's missing alpha 1.
fn widened(part: &str, c: &ir::FragmentColour) -> R<String> {
    if c.components == 4 {
        return Ok(part.to_string());
    }
    let four = vector_name(c.scalar, 4)?;
    let mut parts = Vec::new();
    if c.components == 1 {
        parts.push(part.to_string());
    } else {
        for k in 0..usize::from(c.components) {
            parts.push(format!("{part}.{}", ["x", "y", "z", "w"][k]));
        }
    }
    for k in c.components..4 {
        parts.push(
            match (c.scalar, k) {
                (ir::Scalar::U32, _) => "0u",
                (_, 3) => "1.0",
                _ => "0.0",
            }
            .to_string(),
        );
    }
    Ok(format!("{four}({})", parts.join(", ")))
}

// ---- expressions -------------------------------------------------------------------------------------

fn atom(text: String) -> Val {
    Val { text, atom: true }
}

fn compound(text: String) -> Val {
    Val { text, atom: false }
}

impl FunctionWriter<'_, '_> {
    /// The text of value `v`, defined as `e`; statements it needs first (a level clamped once)
    /// go to `out`.
    fn value(&mut self, v: ir::ValueId, e: &ir::Expr, out: &mut String) -> R<Val> {
        let t = self.f.value_ty(v);
        let m = self.w.m;
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
                let text = literal(c)?;
                if text.starts_with('-') || text.starts_with('(') {
                    compound(text)
                } else {
                    atom(text)
                }
            }
            ir::Expr::Zero(t) => atom(format!("{} {{}}", self.w.ty(*t)?)),
            ir::Expr::Param(i) => {
                if let (
                    Some((map, vt)),
                    Some(ir::EntryPoint { stage: ir::Stage::Fragment { .. }, .. }),
                ) = (self.varyings, self.entry)
                {
                    let mut path = Vec::new();
                    atom(self.rebuild(map, vt, &mut path)?)
                } else {
                    atom(self.params[*i as usize].clone())
                }
            }
            ir::Expr::EntryInput(i) => {
                let arg = match (self.entry, self.position_arg.clone()) {
                    (Some(e), Some(position))
                        if e.inputs.get(*i as usize) == Some(&ir::BuiltinInput::Position) =>
                    {
                        position
                    }
                    _ => self
                        .inputs
                        .get(*i as usize)
                        .cloned()
                        .ok_or("internal: a missing entry input")?,
                };
                let ty = self.w.ty(t)?;
                match m.types.get(t) {
                    ir::TypeDef::Struct { fields, .. } if fields.len() == 3 => {
                        let mut values = Vec::new();
                        for c in ["x", "y", "z"] {
                            values.push(format!("{arg}.{c}"));
                        }
                        atom(format!("{ty} {}", self.w.braced(t, &values)))
                    }
                    _ => atom(format!("{ty} {}", self.w.braced(t, &[arg]))),
                }
            }
            ir::Expr::Load(p) => {
                let (place, atomic) = self.place(p, out)?;
                if atomic {
                    atom(format!(
                        "metal::atomic_load_explicit(&{place}, metal::memory_order_relaxed)"
                    ))
                } else {
                    atom(place)
                }
            }
            ir::Expr::ArrayLength(p) => {
                let (r, elem) = match &p.root {
                    ir::PlaceRoot::Resource(r) if is_sized(&m.resources[r.index()]) => {
                        (r.index(), m.resources[r.index()].ty)
                    }
                    _ => return Err("internal: the length of something not a sized buffer".into()),
                };
                compound(format!("1 + {}", self.max_index(r, elem)?))
            }
            ir::Expr::Texture(op, tex, sampler, args) => {
                self.texture(*op, *tex, *sampler, args, out)?
            }
            ir::Expr::Atomic(op, p, args) => {
                let text = self
                    .atomic(*op, p, args, Some(t), out)?
                    .ok_or("internal: an atomic's value")?;
                atom(text)
            }
            ir::Expr::Barrier => return Err("internal: a barrier's value".into()),
            ir::Expr::Subgroup(op, args) => self.subgroup(*op, args, t)?,
            ir::Expr::TextureStore(..) => return Err("internal: a texel write's value".into()),
            ir::Expr::Discard => return Err("internal: a discard's value".into()),
            ir::Expr::Call(g, args) => {
                if m.functions[g.index()].ret.is_none() {
                    return Err(
                        "internal: a call's value from a function that returns nothing".into()
                    );
                }
                atom(self.call(*g, args)?)
            }
            ir::Expr::Construct(ty, parts) => {
                let texts = parts.iter().map(|p| self.arg(*p)).collect::<R<Vec<_>>>()?;
                let name = self.w.ty(*ty)?;
                match m.types.get(*ty) {
                    ir::TypeDef::Vector(..) | ir::TypeDef::Matrix(..) | ir::TypeDef::Scalar(_) => {
                        atom(format!("{name}({})", texts.join(", ")))
                    }
                    ir::TypeDef::Array(..) => atom(format!("{name} {{{{{}}}}}", texts.join(", "))),
                    _ => atom(format!("{name} {}", self.w.braced(*ty, &texts))),
                }
            }
            ir::Expr::Extract(x, i) => {
                let xt = self.ty_of(*x);
                let base = self.operand(*x)?;
                match m.types.get(xt) {
                    ir::TypeDef::Struct { .. } | ir::TypeDef::Enum { .. } => {
                        let member = self.w.member(xt, *i);
                        let (name, packed) = self.w.member_name(xt, member)?;
                        if packed {
                            atom(format!("{}({base}.{name})", self.w.ty(t)?))
                        } else {
                            atom(format!("{base}.{name}"))
                        }
                    }
                    ir::TypeDef::Vector(..) => {
                        atom(format!("{base}.{}", ["x", "y", "z", "w"][*i as usize]))
                    }
                    ir::TypeDef::Matrix(..) => atom(format!("{base}[{i}]")),
                    ir::TypeDef::Array(..) => atom(format!("{base}.inner[{i}]")),
                    _ => return Err("internal: a part of a value that has none".into()),
                }
            }
            ir::Expr::Variant(et, k, payload) => {
                let ir::TypeDef::Enum { variants, .. } = m.types.get(*et) else {
                    return Err("internal: a variant of a type that isn't an enum".into());
                };
                let name = self.w.ty(*et)?;
                let mut values = vec![format!("{}u", m.types.tag(*et, *k))];
                for (vk, (_, p)) in variants.iter().enumerate() {
                    let Some(pt) = p else { continue };
                    values.push(match payload {
                        Some(x) if vk == *k as usize => self.arg(*x)?,
                        _ => format!("{} {{}}", self.w.ty(*pt)?),
                    });
                }
                atom(format!("{name} {}", self.w.braced(*et, &values)))
            }
            ir::Expr::ExtractDyn(x, i) => {
                let xt = self.ty_of(*x);
                let base = self.operand(*x)?;
                let index = self.index_text(*i, static_len(&m.types, xt), None)?;
                let inner =
                    if matches!(m.types.get(xt), ir::TypeDef::Array(..)) { ".inner" } else { "" };
                atom(format!("{base}{inner}[{index}]"))
            }
            ir::Expr::Splat(x, n) => {
                let s =
                    m.types.as_scalar(self.ty_of(*x)).ok_or("internal: a splat of a non-scalar")?;
                atom(format!("{}({})", vector_name(s, *n)?, self.arg(*x)?))
            }
            ir::Expr::Swizzle(x, comps) => {
                let pattern: String =
                    comps.iter().map(|c| ["x", "y", "z", "w"][*c as usize]).collect();
                atom(format!("{}.{pattern}", self.operand(*x)?))
            }
            ir::Expr::Convert(x, s) => {
                let from = self.scalar_of(*x).ok_or("internal: a conversion of a non-number")?;
                let xt = self.ty_of(*x);
                let to = self.numeric_like(xt, *s)?;
                if from.is_float() && matches!(s, ir::Scalar::I32 | ir::Scalar::U32) {
                    let from_name = self.numeric_like(xt, from)?;
                    let signed = *s == ir::Scalar::I32;
                    self.w.helpers.insert(Helper::F2I(from_name, to.clone(), signed));
                    let name = if signed { "naga_f2i32" } else { "naga_f2u32" };
                    atom(format!("{name}({})", self.arg(*x)?))
                } else {
                    atom(format!("static_cast<{to}>({})", self.arg(*x)?))
                }
            }
            ir::Expr::Bitcast(x, s) => {
                let to = self.numeric_like(self.ty_of(*x), *s)?;
                atom(format!("as_type<{to}>({})", self.arg(*x)?))
            }
            ir::Expr::Select { cond, if_true, if_false } => {
                let vector_cond = matches!(m.types.get(self.ty_of(*cond)), ir::TypeDef::Vector(..));
                if vector_cond {
                    atom(format!(
                        "metal::select({}, {}, {})",
                        self.arg(*if_false)?,
                        self.arg(*if_true)?,
                        self.arg(*cond)?
                    ))
                } else {
                    compound(format!(
                        "{} ? {} : {}",
                        self.operand(*cond)?,
                        self.operand(*if_true)?,
                        self.operand(*if_false)?
                    ))
                }
            }
            ir::Expr::Unary(op, x) => {
                let xt = self.ty_of(*x);
                match (op, m.types.get(xt)) {
                    (ir::UnOp::Neg, ir::TypeDef::Matrix(..)) => {
                        compound(format!("{} * -1.0", self.operand(*x)?))
                    }
                    (ir::UnOp::Neg, _) if self.scalar_of(*x) == Some(ir::Scalar::I32) => {
                        let tn = self.numeric_like(xt, ir::Scalar::I32)?;
                        let un = self.numeric_like(xt, ir::Scalar::U32)?;
                        self.w.helpers.insert(Helper::Neg(tn, un));
                        atom(format!("naga_neg({})", self.arg(*x)?))
                    }
                    (ir::UnOp::Neg, _) => compound(format!("-({})", self.arg(*x)?)),
                    (ir::UnOp::Not, ir::TypeDef::Scalar(ir::Scalar::Bool)) => {
                        compound(format!("!({})", self.arg(*x)?))
                    }
                    (ir::UnOp::Not, _) => compound(format!("~({})", self.arg(*x)?)),
                }
            }
            ir::Expr::Binary(op, a, b) => self.binary(*op, *a, *b, t)?,
            ir::Expr::Builtin(b, args) => self.builtin(*b, args, t)?,
            ir::Expr::Host(..) => return Err("GPU code can't record GPU work".into()),
            ir::Expr::Mem(..) => return Err("GPU code can't use raw memory".into()),
            ir::Expr::Run(_) | ir::Expr::Addr(_) => {
                return Err("internal: a CPU-only expression in GPU code".into());
            }
        })
    }

    /// The part at `path` of a fragment shader's input, of IR type `t`, rebuilt from the members
    /// of its input struct (and its position argument).
    fn rebuild(&mut self, map: &[ir::Varying], t: ir::TypeId, path: &mut Vec<u32>) -> R<String> {
        if let Some(k) = map.iter().position(|v| v.path == *path) {
            return Ok(if map[k].position {
                self.position_arg.clone().ok_or("internal: no position argument")?
            } else {
                format!("varyings.v{k}")
            });
        }
        let types = &self.w.m.types;
        let n = match types.get(t) {
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
        let ty = self.w.ty(t)?;
        Ok(match self.w.m.types.get(t) {
            ir::TypeDef::Matrix(..) => format!("{ty}({})", parts.join(", ")),
            _ => format!("{ty} {}", self.w.braced(t, &parts)),
        })
    }

    fn binary(&mut self, op: ir::BinOp, a: ir::ValueId, b: ir::ValueId, t: ir::TypeId) -> R<Val> {
        let m = self.w.m;
        let ta = self.ty_of(a);
        let scalar = self.scalar_of(a);
        let is_bool = matches!(m.types.get(ta), ir::TypeDef::Scalar(ir::Scalar::Bool));
        let is_int = matches!(
            m.types.get(ta),
            ir::TypeDef::Scalar(ir::Scalar::I32 | ir::Scalar::U32)
                | ir::TypeDef::Vector(ir::Scalar::I32 | ir::Scalar::U32, _)
        );
        let signed = scalar == Some(ir::Scalar::I32) && is_int;
        let (l, r) = (self.operand(a)?, self.operand(b)?);
        use ir::BinOp as B;
        if is_int
            && matches!(op, B::Div | B::Rem)
            && self.w.checks.int_div
            && !self.w.proofs.safe_divisor(self.f, b)
        {
            let tn = self.w.ty(t)?;
            let helper =
                if op == B::Div { Helper::Div(tn, signed) } else { Helper::Mod(tn, signed) };
            self.w.helpers.insert(helper);
            let name = if op == B::Div { "naga_div" } else { "naga_mod" };
            return Ok(atom(format!("{name}({}, {})", self.arg(a)?, self.arg(b)?)));
        }
        if op == B::Rem && scalar.is_some_and(|s| s.is_float()) {
            return Ok(atom(format!("metal::fmod({}, {})", self.arg(a)?, self.arg(b)?)));
        }
        let sym = match op {
            B::Add | B::WrappingAdd => "+",
            B::Sub | B::WrappingSub => "-",
            B::Mul | B::WrappingMul => "*",
            B::Div => "/",
            B::Rem => "%",
            B::Shl => "<<",
            B::Shr => ">>",
            B::BitAnd => "&",
            B::BitOr => "|",
            B::BitXor if is_bool => "!=",
            B::BitXor => "^",
            B::And if is_bool => "&&",
            B::And => "&",
            B::Or if is_bool => "||",
            B::Or => "|",
            B::Eq => "==",
            B::Ne => "!=",
            B::Lt => "<",
            B::Le => "<=",
            B::Gt => ">",
            B::Ge => ">=",
        };
        // Metal's compiler, even in relaxed math, may fold a float's equality as if neither side
        // were a NaN (`x != x` false), but keeps `isnan` and `isunordered`, which say it: a float
        // compared with itself is `isnan`, and any other equality says whether the sides are
        // unordered too. (Ordered comparisons with a NaN are left as WGSL leaves them.)
        if scalar.is_some_and(|s| s.is_float()) && matches!(op, B::Eq | B::Ne) {
            let x = self.arg(a)?;
            if a == b {
                return Ok(if op == B::Ne {
                    atom(format!("metal::isnan({x})"))
                } else {
                    compound(format!("!metal::isnan({x})"))
                });
            }
            let y = self.arg(b)?;
            let vector = matches!(m.types.get(ta), ir::TypeDef::Vector(..));
            let (and, or) = if vector { ("&", "|") } else { ("&&", "||") };
            return Ok(atom(if op == B::Ne {
                format!("(({l} != {r}) {or} metal::isunordered({x}, {y}))")
            } else {
                format!("(({l} == {r}) {and} !metal::isunordered({x}, {y}))")
            }));
        }
        // Signed arithmetic that may overflow is done unsigned, as WGSL defines it (it wraps).
        if signed
            && matches!(
                op,
                B::Add | B::Sub | B::Mul | B::WrappingAdd | B::WrappingSub | B::WrappingMul
            )
        {
            let tn = self.w.ty(t)?;
            let un = self.numeric_like(t, ir::Scalar::U32)?;
            let ua = format!("as_type<{un}>({})", self.arg(a)?);
            let ub = format!("as_type<{un}>({})", self.arg(b)?);
            return Ok(atom(format!("as_type<{tn}>({ua} {sym} {ub})")));
        }
        Ok(compound(format!("{l} {sym} {r}")))
    }

    fn builtin(&mut self, b: ir::Builtin, args: &[ir::ValueId], t: ir::TypeId) -> R<Val> {
        use ir::Builtin as B;
        let m = self.w.m;
        let texts = args.iter().map(|a| self.arg(*a)).collect::<R<Vec<_>>>()?;
        let a0 = args.first().copied().ok_or("internal: a builtin with no argument")?;
        let at = self.ty_of(a0);
        let scalar_arg = matches!(m.types.get(at), ir::TypeDef::Scalar(_));
        let int_kind = self.scalar_of(a0);
        let call = |name: &str| atom(format!("metal::{name}({})", texts.join(", ")));
        Ok(match b {
            B::Dpdx => call("dfdx"),
            B::Dpdy => call("dfdy"),
            B::Fwidth => call("fwidth"),
            B::AllEqual => {
                atom(format!("metal::all({} == {})", self.operand(a0)?, self.operand(args[1])?))
            }
            B::CountOnes | B::LeadingZeros | B::TrailingZeros => {
                let name = match b {
                    B::CountOnes => "popcount",
                    B::LeadingZeros => "clz",
                    _ => "ctz",
                };
                let count = format!("metal::{name}({})", texts[0]);
                if int_kind == Some(ir::Scalar::U32) {
                    atom(count)
                } else {
                    let to = self.numeric_like(at, ir::Scalar::U32)?;
                    atom(format!("as_type<{to}>({count})"))
                }
            }
            B::Abs if int_kind == Some(ir::Scalar::I32) => {
                let tn = self.numeric_like(at, ir::Scalar::I32)?;
                let un = self.numeric_like(at, ir::Scalar::U32)?;
                self.w.helpers.insert(Helper::Abs(tn, un));
                atom(format!("naga_abs({})", texts[0]))
            }
            B::Sign if int_kind == Some(ir::Scalar::I32) => {
                let (minus, plus) = match m.types.get(at) {
                    ir::TypeDef::Vector(_, n) => (format!("int{n}(-1)"), format!("int{n}(1)")),
                    _ => ("int(-1)".to_string(), "int(1)".to_string()),
                };
                let x = &texts[0];
                atom(format!(
                    "metal::select(metal::select({minus}, {plus}, ({x} > 0)), int(0), ({x} == 0))"
                ))
            }
            B::Length if scalar_arg => call("abs"),
            B::Distance if scalar_arg => {
                atom(format!("metal::abs({} - {})", self.operand(a0)?, self.operand(args[1])?))
            }
            B::Dot if int_kind.is_some_and(|s| !s.is_float()) => {
                let ir::TypeDef::Vector(s, n) = *m.types.get(at) else {
                    return Err("internal: a dot of non-vectors".into());
                };
                let vec = vector_name(s, n)?;
                let sn = scalar_name(s)?.to_string();
                self.w.helpers.insert(Helper::Dot(vec, sn.clone(), n));
                atom(format!("naga_dot_{sn}{n}({}, {})", texts[0], texts[1]))
            }
            _ => {
                let _ = t;
                let name = match b {
                    B::Sqrt => "sqrt",
                    B::InverseSqrt => "rsqrt",
                    B::Sin => "sin",
                    B::Cos => "cos",
                    B::Tan => "tan",
                    B::Asin => "asin",
                    B::Acos => "acos",
                    B::Atan => "atan",
                    B::Atan2 => "atan2",
                    B::Exp => "exp",
                    B::Exp2 => "exp2",
                    B::Log => "log",
                    B::Log2 => "log2",
                    B::Sinh => "sinh",
                    B::Cosh => "cosh",
                    B::Tanh => "tanh",
                    B::Pow => "pow",
                    B::Floor => "floor",
                    B::Ceil => "ceil",
                    B::Round => "rint",
                    B::Trunc => "trunc",
                    B::Fract => "fract",
                    B::Abs => "abs",
                    B::Sign => "sign",
                    B::Min => "min",
                    B::Max => "max",
                    B::Clamp => "clamp",
                    B::Saturate => "saturate",
                    B::Mix => "mix",
                    B::Step => "step",
                    B::Smoothstep => "smoothstep",
                    B::Length => "length",
                    B::Distance => "distance",
                    B::Dot => "dot",
                    B::Cross => "cross",
                    B::Normalize => "normalize",
                    B::Dpdx
                    | B::Dpdy
                    | B::Fwidth
                    | B::AllEqual
                    | B::CountOnes
                    | B::LeadingZeros
                    | B::TrailingZeros => unreachable!("handled above"),
                };
                call(name)
            }
        })
    }

    /// An atomic operation's call (`ir::Expr::Atomic`); with `want`, as a value of that type (a
    /// compare-exchange's: a struct of the old value and whether it stored). A store is
    /// written to `out` and has no value.
    fn atomic(
        &mut self,
        op: ir::AtomicOp,
        p: &ir::Place,
        args: &[ir::ValueId],
        want: Option<ir::TypeId>,
        out: &mut String,
    ) -> R<Option<String>> {
        use ir::AtomicOp as A;
        let m = self.w.m;
        let pt = m.place_ty(self.f, p).ok_or("internal: an atomic place's type")?;
        let ir::TypeDef::Atomic(s) = *m.types.get(pt) else {
            return Err("internal: an atomic operation on something that isn't atomic".into());
        };
        let (place, _) = self.place(p, out)?;
        if op == A::Load {
            return Ok(Some(format!(
                "metal::atomic_load_explicit(&{place}, metal::memory_order_relaxed)"
            )));
        }
        let first = self.arg(*args.first().ok_or("internal: an atomic without an operand")?)?;
        if op == A::Store {
            let _ = writeln!(
                out,
                "metal::atomic_store_explicit(&{place}, {first}, metal::memory_order_relaxed);"
            );
            return Ok(None);
        }
        if op == A::CompareExchange {
            let value = self
                .arg(*args.get(1).ok_or("internal: a compare-exchange without its new value")?)?;
            let sn = scalar_name(s)?.to_string();
            let result = format!("exchange_{sn}");
            self.w.exchange_results.insert(sn.clone());
            self.w.helpers.insert(Helper::CompareExchange(sn, result.clone()));
            let call =
                format!("naga_atomic_compare_exchange_weak_explicit(&{place}, {first}, {value})");
            let Some(t) = want else { return Ok(Some(call)) };
            // The old value and whether it stored, as the IR's struct, its `bool` as a `u32` if
            // it's a type memory holds.
            let tmp = self.fresh("_exchanged");
            let _ = writeln!(out, "{result} {tmp} = {call};");
            let types = &m.types;
            let stored =
                if types.field(t, 1).and_then(|b| types.as_scalar(b)) == Some(ir::Scalar::U32) {
                    format!("static_cast<uint>({tmp}.exchanged)")
                } else {
                    format!("{tmp}.exchanged")
                };
            let ty = self.w.ty(t)?;
            let braced = self.w.braced(t, &[format!("{tmp}.old_value"), stored]);
            return Ok(Some(format!("{ty} {braced}")));
        }
        let fun = match op {
            A::Add => "fetch_add",
            A::Sub => "fetch_sub",
            A::Min => "fetch_min",
            A::Max => "fetch_max",
            A::And => "fetch_and",
            A::Or => "fetch_or",
            A::Xor => "fetch_xor",
            A::Exchange => "exchange",
            A::Load | A::Store | A::CompareExchange => unreachable!("handled above"),
        };
        Ok(Some(format!(
            "metal::atomic_{fun}_explicit(&{place}, {first}, metal::memory_order_relaxed)"
        )))
    }

    /// A subgroup operation's value.
    fn subgroup(&mut self, op: ir::SubgroupOp, args: &[ir::ValueId], t: ir::TypeId) -> R<Val> {
        use ir::SubgroupOp as S;
        let _ = t;
        let x =
            self.arg(*args.first().ok_or("internal: a subgroup operation without its operand")?)?;
        let name = match op {
            S::Ballot => {
                return Ok(atom(format!(
                    "metal::uint4((uint64_t)metal::simd_ballot({x}), 0, 0, 0)"
                )));
            }
            S::Add => "simd_sum",
            S::ExclusiveAdd => "simd_prefix_exclusive_sum",
            S::InclusiveAdd => "simd_prefix_inclusive_sum",
            S::Min => "simd_min",
            S::Max => "simd_max",
            S::And => "simd_and",
            S::Or => "simd_or",
            S::Xor => "simd_xor",
            S::Any => "simd_any",
            S::All => "simd_all",
            S::BroadcastFirst => "simd_broadcast_first",
            S::Shuffle => {
                let from =
                    self.arg(*args.get(1).ok_or("internal: a shuffle without its source")?)?;
                return Ok(atom(format!("metal::simd_shuffle({x}, {from})")));
            }
        };
        Ok(atom(format!("metal::{name}({x})")))
    }

    /// A texture read (`ir::Expr::Texture`).
    fn texture(
        &mut self,
        op: ir::TextureOp,
        tex: ir::ResourceId,
        sampler: Option<ir::ResourceId>,
        args: &[ir::ValueId],
        out: &mut String,
    ) -> R<Val> {
        use ir::TextureOp as T;
        let m = self.w.m;
        let image = self.w.resource_names[tex.index()].clone();
        let (dim, depth) = match m.resources[tex.index()].kind {
            ir::ResourceKind::Texture { dim, depth, .. } => (dim, depth),
            _ => return Err("internal: a texture read of something that isn't a texture".into()),
        };
        let arg = |fw: &Self, i: usize| -> R<String> {
            fw.arg(*args.get(i).ok_or("internal: a texture read without its arguments")?)
        };
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
                let sampler = self.w.resource_names[s.index()].clone();
                let main = match op {
                    T::Gather(_) => "gather",
                    T::GatherCompare => "gather_compare",
                    T::SampleCompare | T::SampleCompareLevel => "sample_compare",
                    _ => "sample",
                };
                let mut parts = vec![sampler, arg(self, 0)?];
                if let Some(l) = &layer {
                    parts.push(l.clone());
                }
                match op {
                    T::SampleCompare | T::SampleCompareLevel | T::GatherCompare => {
                        parts.push(arg(self, rest)?);
                    }
                    _ => {}
                }
                match op {
                    T::SampleLevel => parts.push(format!("metal::level({})", arg(self, rest)?)),
                    T::SampleBias => parts.push(format!("metal::bias({})", arg(self, rest)?)),
                    T::SampleGrad => {
                        let g = match dim {
                            ir::TexDim::D3 => "gradient3d",
                            ir::TexDim::Cube => "gradientcube",
                            _ => "gradient2d",
                        };
                        parts.push(format!(
                            "metal::{g}({}, {})",
                            arg(self, rest)?,
                            arg(self, rest + 1)?
                        ));
                    }
                    T::Gather(c) if c != 0 => {
                        if dim != ir::TexDim::Cube {
                            parts.push("metal::int2(0)".into());
                        }
                        parts.push(format!(
                            "metal::component::{}",
                            ["x", "y", "z", "w"][c as usize]
                        ));
                    }
                    _ => {}
                }
                atom(format!("{image}.{main}({})", parts.join(", ")))
            }
            T::Load => {
                let n = dim.texel();
                let coords = (0..n).map(|i| arg(self, i)).collect::<R<Vec<_>>>()?;
                let level = arg(self, rest)?;
                let uvec = format!("metal::uint{n}({})", coords.join(", "));
                if !self.w.checks.bounds {
                    let mut parts = vec![uvec];
                    if let Some(l) = &layer {
                        parts.push(l.clone());
                    }
                    parts.push(level);
                    return Ok(atom(format!("{image}.read({})", parts.join(", "))));
                }
                // The level clamped once, before the read; then the texel and the layer.
                let lod = self.fresh("clamped_lod_e");
                let _ = writeln!(
                    out,
                    "uint {lod} = metal::min(uint({level}), {image}.get_num_mip_levels() - 1);"
                );
                let size = match n {
                    3 => format!(
                        "metal::uint3({image}.get_width({lod}), {image}.get_height({lod}), {image}.get_depth({lod}))"
                    ),
                    _ => {
                        format!("metal::uint2({image}.get_width({lod}), {image}.get_height({lod}))")
                    }
                };
                let mut parts = vec![format!("metal::min({uvec}, {size} - 1)")];
                if let Some(l) = &layer {
                    parts.push(format!("metal::min(uint({l}), {image}.get_array_size() - 1)"));
                }
                parts.push(lod);
                let _ = depth;
                atom(format!("{image}.read({})", parts.join(", ")))
            }
            T::Width => atom(format!("{image}.get_width()")),
            T::Height => atom(format!("{image}.get_height()")),
            T::Depth => atom(format!("{image}.get_depth()")),
            T::Layers => atom(format!("{image}.get_array_size()")),
            T::Levels => atom(format!("{image}.get_num_mip_levels()")),
        })
    }

    /// A texel's write (`ir::Expr::TextureStore`).
    fn texture_store(
        &mut self,
        tex: ir::ResourceId,
        args: &[ir::ValueId],
        out: &mut String,
    ) -> R<()> {
        let m = self.w.m;
        let image = self.w.resource_names[tex.index()].clone();
        let Some((&value, xs)) = args.split_last() else {
            return Err("internal: a texel's write without its arguments".into());
        };
        let arrayed = matches!(
            m.resources[tex.index()].kind,
            ir::ResourceKind::StorageTexture { dim, .. } if dim.arrayed()
        );
        let (xs, layer) = match (arrayed, xs.split_last()) {
            (true, Some((&l, xs))) => (xs, Some(self.arg(l)?)),
            _ => (xs, None),
        };
        let coords = xs.iter().map(|x| self.arg(*x)).collect::<R<Vec<_>>>()?;
        let mut parts =
            vec![self.arg(value)?, format!("metal::uint{}({})", coords.len(), coords.join(", "))];
        if let Some(l) = layer {
            parts.push(l);
        }
        let _ = writeln!(out, "{image}.write({});", parts.join(", "));
        Ok(())
    }
}
