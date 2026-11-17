//! The host tests' shaders, each built as IR (`wrela_shader::build`) and written for this
//! machine's target by the compiler's own writers: what WGSL each was before is in its comment.

use wrela_ir::{Expr, Proj};
use wrela_shader::build::{
    AtomicOp, BinOp, Body, Builder, BuiltinInput, Scalar, TexDim, TextureFormat, ValueId,
};

/// `vec4f(x, y, z, w)` of constants.
fn vec4(f: &mut Body<'_>, xyzw: [f32; 4]) -> ValueId {
    let parts: Vec<ValueId> = xyzw.iter().map(|x| f.f32(*x)).collect();
    let v4 = f.types().vector(4);
    f.construct(v4, &parts)
}

/// Element `i` of a constant array of `vec2f`s: `array(vec2f(..), ..)[i]`.
fn corner(f: &mut Body<'_>, corners: &[[f32; 2]], i: ValueId) -> ValueId {
    let v2 = f.types().vector(2);
    let parts: Vec<ValueId> = corners
        .iter()
        .map(|[x, y]| {
            let (x, y) = (f.f32(*x), f.f32(*y));
            f.construct(v2, &[x, y])
        })
        .collect();
    let arr = f.types().intern(wrela_ir::TypeDef::Array(v2, corners.len() as u32));
    let a = f.construct(arr, &parts);
    f.extract_dyn(v2, a, i)
}

/// `vec4f(xy, z, w)`.
fn widen(f: &mut Body<'_>, xy: ValueId, z: ValueId, w: ValueId) -> ValueId {
    let fl = f.types().f32();
    let x = f.extract(fl, xy, 0);
    let y = f.extract(fl, xy, 1);
    let v4 = f.types().vector(4);
    f.construct(v4, &[x, y, z, w])
}

/// The vertex index as the shader takes it.
fn vertex_index(f: &mut Body<'_>, k: u32) -> ValueId {
    let u = f.types().u32();
    let s = f.types().intern(wrela_ir::TypeDef::Struct {
        name: "VertexIndex".into(),
        fields: vec![("index".into(), u)],
    });
    let v = f.input(k, s);
    f.extract(u, v, 0)
}

/// ```text
/// struct P { index: u32, value: u32, _pad: vec2u }
/// @group(0) @binding(0) var<uniform> p: P;
/// @group(0) @binding(1) var<storage, read_write> out: array<u32>;
/// @compute @workgroup_size(1) fn store() { out[p.index] = p.value; }
/// ```
pub fn store() -> Builder {
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let u2 = b.vector(Scalar::U32, 2);
    let p_ty = b.structure("P", &[("index", u), ("value", u), ("_pad", u2)]);
    let p = b.uniform("p", 0, p_ty);
    let out = b.storage("out", 1, u, true);
    b.compute("store", [1, 1, 1], &[], |f| {
        let index = f.load(u, Builder::whole(p).with(Proj::Field(0)));
        let value = f.load(u, Builder::whole(p).with(Proj::Field(1)));
        f.store(Builder::element(out, index), value);
    });
    b
}

/// The cell pipeline: fills the pixel at `cell` of a screen of `size` with `colour` (or with
/// `fill`, a constant):
/// ```text
/// struct C { cell: vec2u, size: vec2u, colour: u32, _pad: u32, _pad2: vec2u }
/// @group(0) @binding(0) var<uniform> c: C;
/// @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
///     let corner = array(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1), vec2f(1, 0), vec2f(1, 1), vec2f(0, 1))[i];
///     let px = (vec2f(c.cell) + corner) / vec2f(c.size);
///     return vec4f(px.x * 2 - 1, 1 - px.y * 2, 0, 1);
/// }
/// @fragment fn fs() -> @location(0) vec4f { return unpack4x8unorm(c.colour); }
/// ```
pub fn cell(fill: Option<[f32; 4]>) -> Builder {
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let u2 = b.vector(Scalar::U32, 2);
    let v2 = b.vector(Scalar::F32, 2);
    let fl = b.scalar(Scalar::F32);
    let c_ty =
        b.structure("C", &[("cell", u2), ("size", u2), ("colour", u), ("_pad", u), ("_pad2", u2)]);
    let c = b.uniform("c", 0, c_ty);
    b.vertex("vs", &[BuiltinInput::VertexIndex], |f| {
        let i = vertex_index(f, 0);
        let corners = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let corner = corner(f, &corners, i);
        let cell = f.load(u2, Builder::whole(c).with(Proj::Field(0)));
        let size = f.load(u2, Builder::whole(c).with(Proj::Field(1)));
        let cellf = f.convert(v2, cell, Scalar::F32);
        let sizef = f.convert(v2, size, Scalar::F32);
        let at = f.binary(BinOp::Add, cellf, corner);
        let px = f.binary(BinOp::Div, at, sizef);
        let (x, y) = (f.extract(fl, px, 0), f.extract(fl, px, 1));
        let (two, one, zero) = (f.f32(2.0), f.f32(1.0), f.f32(0.0));
        let x2 = f.binary(BinOp::Mul, x, two);
        let cx = f.binary(BinOp::Sub, x2, one);
        let y2 = f.binary(BinOp::Mul, y, two);
        let cy = f.binary(BinOp::Sub, one, y2);
        let v4 = f.types().vector(4);
        let p = f.construct(v4, &[cx, cy, zero, one]);
        f.ret_clip(p);
    });
    b.fragment("fs", &[], |f| {
        let colour = match fill {
            Some(c) => vec4(f, c),
            None => {
                let packed = f.load(u, Builder::whole(c).with(Proj::Field(2)));
                f.unpack4x8unorm(packed)
            }
        };
        f.ret(Some(colour));
    });
    b
}

/// ```text
/// @group(0) @binding(0) var<storage, read> a: array<u32>;
/// @group(0) @binding(1) var<storage, read> b: array<u32>;
/// @group(0) @binding(2) var<storage, read_write> out: array<u32>;
/// @compute @workgroup_size(1) fn sum() { out[0] += a[0] + b[0]; }
/// ```
pub fn sum() -> Builder {
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let a = b.storage("a", 0, u, false);
    let bb = b.storage("b", 1, u, false);
    let out = b.storage("out", 2, u, true);
    b.compute("sum", [1, 1, 1], &[], |f| {
        let zero = f.u32(0);
        let x = f.load(u, Builder::element(a, zero));
        let y = f.load(u, Builder::element(bb, zero));
        let s = f.binary(BinOp::Add, x, y);
        let old = f.load(u, Builder::element(out, zero));
        let new = f.binary(BinOp::Add, old, s);
        f.store(Builder::element(out, zero), new);
    });
    b
}

/// ```text
/// @group(0) @binding(0) var source: texture_2d<f32>;
/// @group(0) @binding(1) var<storage, read_write> out: array<f32>;
/// @compute @workgroup_size(1) fn read_texel() { out[0] = textureLoad(source, vec2i(0, 0), 0).x; }
/// ```
pub fn read_texel() -> Builder {
    let mut b = Builder::new();
    let fl = b.scalar(Scalar::F32);
    let source = b.texture("source", 0, TexDim::D2, TextureFormat::R16Float, false);
    let out = b.storage("out", 1, fl, true);
    b.compute("read_texel", [1, 1, 1], &[], |f| {
        let zero = f.u32(0);
        let t = f.texel(source, zero, zero, zero);
        let x = f.extract(fl, t, 0);
        f.store(Builder::element(out, zero), x);
    });
    b
}

/// ```text
/// @group(0) @binding(0) var<storage, read_write> flag: atomic<u32>;
/// @compute @workgroup_size(1) fn flag_error() { atomicStore(&flag, 1u); }
/// ```
/// (its flag the first of an array of them).
pub fn flag() -> Builder {
    let mut b = Builder::new();
    let au = b.atomic(Scalar::U32);
    let u = b.scalar(Scalar::U32);
    let flag = b.storage("flag", 0, au, true);
    b.compute("flag_error", [1, 1, 1], &[], |f| {
        let zero = f.u32(0);
        let one = f.u32(1);
        f.atomic(u, AtomicOp::Store, Builder::element(flag, zero), &[one]);
    });
    b
}

/// One triangle covering the screen:
/// ```text
/// @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
///     let corners = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
///     return vec4f(corners[i], 0, 1);
/// }
/// ```
fn covering(b: &mut Builder) {
    b.vertex("vs", &[BuiltinInput::VertexIndex], |f| {
        let i = vertex_index(f, 0);
        let xy = corner(f, &[[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]], i);
        let (zero, one) = (f.f32(0.0), f.f32(1.0));
        let p = widen(f, xy, zero, one);
        f.ret_clip(p);
    });
}

/// The covering triangle, white: `@fragment fn fs() -> @location(0) vec4f { return vec4f(1, 1, 1, 1); }`.
pub fn fill() -> Builder {
    let mut b = Builder::new();
    covering(&mut b);
    b.fragment("fs", &[], |f| {
        let white = vec4(f, [1.0, 1.0, 1.0, 1.0]);
        f.ret(Some(white));
    });
    b
}

/// The covering triangle coloured by a buffer it reads:
/// `@fragment fn fs() -> @location(0) vec4f { return unpack4x8unorm(data[0]); }`.
pub fn read_pass() -> Builder {
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let data = b.storage("data", 0, u, false);
    covering(&mut b);
    b.fragment("fs", &[], |f| {
        let zero = f.u32(0);
        let x = f.load(u, Builder::element(data, zero));
        let c = f.unpack4x8unorm(x);
        f.ret(Some(c));
    });
    b
}

/// The covering triangle writing the buffer it's given:
/// `@fragment fn fs() -> @location(0) vec4f { atomicStore(&data[0], 42u); return vec4f(0, 1, 0, 1); }`.
pub fn write_pass() -> Builder {
    let mut b = Builder::new();
    let au = b.atomic(Scalar::U32);
    let u = b.scalar(Scalar::U32);
    let data = b.storage("data", 0, au, true);
    covering(&mut b);
    b.fragment("fs", &[], |f| {
        let zero = f.u32(0);
        let v = f.u32(42);
        f.atomic(u, AtomicOp::Store, Builder::element(data, zero), &[v]);
        let green = vec4(f, [0.0, 1.0, 0.0, 1.0]);
        f.ret(Some(green));
    });
    b
}

/// ```text
/// struct M { colour: u32, z: f32, _pad: vec2u }
/// @group(0) @binding(0) var<uniform> m: M;
/// @group(0) @binding(1) var<storage, read> pos: array<vec2f>;
/// @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(pos[i], m.z, 1); }
/// @fragment fn fs() -> @location(0) vec4f { return unpack4x8unorm(m.colour); }
/// ```
pub fn mesh() -> Builder {
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let u2 = b.vector(Scalar::U32, 2);
    let fl = b.scalar(Scalar::F32);
    let v2 = b.vector(Scalar::F32, 2);
    let m_ty = b.structure("M", &[("colour", u), ("z", fl), ("_pad", u2)]);
    let m = b.uniform("m", 0, m_ty);
    let pos = b.storage("pos", 1, v2, false);
    b.vertex("vs", &[BuiltinInput::VertexIndex], |f| {
        let i = vertex_index(f, 0);
        let xy = f.load(v2, Builder::element(pos, i));
        let z = f.load(fl, Builder::whole(m).with(Proj::Field(1)));
        let one = f.f32(1.0);
        let p = widen(f, xy, z, one);
        f.ret_clip(p);
    });
    b.fragment("fs", &[], |f| {
        let packed = f.load(u, Builder::whole(m).with(Proj::Field(0)));
        let c = f.unpack4x8unorm(packed);
        f.ret(Some(c));
    });
    b
}

/// ```text
/// @vertex fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
///     -> @builtin(position) vec4f {
///     let corners = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
///     let position = corners[vertex] + vec2f(f32(instance) - 5.0, 0.0) * 10.0;
///     return vec4f(position, 0, 1);
/// }
/// @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
/// ```
pub fn instance() -> Builder {
    let mut b = Builder::new();
    let fl = b.scalar(Scalar::F32);
    let v2 = b.vector(Scalar::F32, 2);
    let u = b.scalar(Scalar::U32);
    b.vertex("vs", &[BuiltinInput::VertexIndex, BuiltinInput::InstanceIndex], |f| {
        let vertex = vertex_index(f, 0);
        let s = f.types().intern(wrela_ir::TypeDef::Struct {
            name: "InstanceIndex".into(),
            fields: vec![("index".into(), u)],
        });
        let inst = f.input(1, s);
        let instance = f.extract(u, inst, 0);
        let c = corner(f, &[[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]], vertex);
        let fi = f.convert(fl, instance, Scalar::F32);
        let five = f.f32(5.0);
        let dx = f.binary(BinOp::Sub, fi, five);
        let zero = f.f32(0.0);
        let d = f.construct(v2, &[dx, zero]);
        let ten = f.f32(10.0);
        let tens = f.let_(v2, Expr::Splat(ten, 2));
        let shift = f.binary(BinOp::Mul, d, tens);
        let at = f.binary(BinOp::Add, c, shift);
        let one = f.f32(1.0);
        let p = widen(f, at, zero, one);
        f.ret_clip(p);
    });
    b.fragment("fs", &[], |f| {
        let green = vec4(f, [0.0, 1.0, 0.0, 1.0]);
        f.ret(Some(green));
    });
    b
}

/// What a NaN and a sum's order come to:
/// ```text
/// @group(0) @binding(0) var<storage, read> input: array<f32>;
/// @group(0) @binding(1) var<storage, read_write> out: array<u32>;
/// @compute @workgroup_size(1) fn probe() {
///     let zero = input[0]; let big = input[1]; let one = input[2];
///     let nan = zero / zero;
///     out[0] = select(0u, 1u, nan != nan);
///     out[1] = bitcast<u32>((big + one) - big);
///     out[2] = bitcast<u32>(input[3] * input[4] + input[5]);
/// }
/// ```
#[cfg(target_vendor = "apple")]
pub fn probe() -> Builder {
    let mut b = Builder::new();
    let fl = b.scalar(Scalar::F32);
    let u = b.scalar(Scalar::U32);
    let input = b.storage("input", 0, fl, false);
    let out = b.storage("out", 1, u, true);
    b.compute("probe", [1, 1, 1], &[], |f| {
        let read = |f: &mut Body<'_>, k: u32| {
            let at = f.u32(k);
            f.load(fl, Builder::element(input, at))
        };
        let (zero, big, one) = (read(f, 0), read(f, 1), read(f, 2));
        let nan = f.binary(BinOp::Div, zero, zero);
        let unequal = f.binary(BinOp::Ne, nan, nan);
        let (u0, u1) = (f.u32(0), f.u32(1));
        let flagged = f.select(unequal, u1, u0);
        let sum = f.binary(BinOp::Add, big, one);
        let back = f.binary(BinOp::Sub, sum, big);
        let bits = f.bitcast(u, back, Scalar::U32);
        let (x, y, z) = (read(f, 3), read(f, 4), read(f, 5));
        let xy = f.binary(BinOp::Mul, x, y);
        let fused = f.binary(BinOp::Add, xy, z);
        let fused_bits = f.bitcast(u, fused, Scalar::U32);
        for (k, v) in [flagged, bits, fused_bits].into_iter().enumerate() {
            let at = f.u32(k as u32);
            f.store(Builder::element(out, at), v);
        }
    });
    b
}

/// first-light's compute pipeline: derives palette entries 4.. from the four seed colours,
/// adding `step` once per group of four (wrapping `u32` arithmetic, so a host can check it
/// exactly).
/// ```text
/// struct Params { step: u32, count: u32, _pad: vec2u }
/// @group(0) @binding(0) var<uniform> params: Params;
/// @group(0) @binding(1) var<storage, read_write> palette: array<u32>;
/// @compute @workgroup_size(64)
/// fn fill(@builtin(global_invocation_id) id: vec3u) {
///     let i = id.x;
///     if i >= 4u && i < params.count {
///         palette[i] = palette[i % 4u] + (i / 4u) * params.step;
///     }
/// }
/// ```
pub fn first_light_fill() -> Builder {
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let u2 = b.vector(Scalar::U32, 2);
    let u3 = b.vector(Scalar::U32, 3);
    let params_ty = b.structure("Params", &[("step", u), ("count", u), ("_pad", u2)]);
    let params = b.uniform("params", 0, params_ty);
    let palette = b.storage("palette", 1, u, true);
    b.compute("fill", [64, 1, 1], &[BuiltinInput::GlobalInvocationId], |f| {
        let id = f.input(0, u3);
        let i = f.extract(u, id, 0);
        let four = f.u32(4);
        let past = f.binary(BinOp::Ge, i, four);
        let count = f.load(u, Builder::whole(params).with(Proj::Field(1)));
        let under = f.binary(BinOp::Lt, i, count);
        let both = f.binary(BinOp::And, past, under);
        f.if_(
            both,
            |f| {
                let seed = f.binary(BinOp::Rem, i, four);
                let base = f.load(u, Builder::element(palette, seed));
                let group = f.binary(BinOp::Div, i, four);
                let step = f.load(u, Builder::whole(params).with(Proj::Field(0)));
                let add = f.binary(BinOp::Mul, group, step);
                let value = f.binary(BinOp::Add, base, add);
                f.store(Builder::element(palette, i), value);
            },
            |_| {},
        );
    });
    b
}

/// first-light's render pipeline: one full-screen triangle; each pixel is coloured by its
/// distance to a circle whose centre the CPU computes each frame, from the palette `fill`
/// derived on the first frame.
/// ```text
/// struct Frame { resolution: vec2f, center: vec2f, time: f32, radius: f32, _pad: vec2f }
/// @group(0) @binding(0) var<uniform> frame: Frame;
/// @group(0) @binding(1) var<storage, read> palette: array<u32>;
/// @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
///     let x = f32(i32(i & 1u) * 4 - 1);
///     let y = f32(i32(i >> 1u) * 4 - 1);
///     return vec4f(x, y, 0.0, 1.0);
/// }
/// fn colour(i: u32) -> vec3f { return unpack4x8unorm(palette[i]).rgb; }
/// @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
///     let p = pos.xy;
///     var c = mix(colour(0u), colour(2u), p.y / frame.resolution.y);
///     let d = distance(p, frame.center) - frame.radius;
///     c += colour(7u) * 0.4 * exp(-max(d, 0.0) / (0.3 * frame.radius));
///     let pulse = 0.85 + 0.15 * cos(frame.time * 3.0);
///     c = mix(colour(5u) * pulse, c, smoothstep(-1.0, 1.0, d));
///     return vec4f(min(c, vec3f(1.0)), 1.0);
/// }
/// ```
pub fn first_light_render() -> Builder {
    use wrela_ir::Builtin as B;
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let i32t = b.scalar(Scalar::I32);
    let fl = b.scalar(Scalar::F32);
    let v2 = b.vector(Scalar::F32, 2);
    let v3 = b.vector(Scalar::F32, 3);
    let v4 = b.vector(Scalar::F32, 4);
    let frame_ty = b.structure(
        "Frame",
        &[("resolution", v2), ("center", v2), ("time", fl), ("radius", fl), ("_pad", v2)],
    );
    let frame = b.uniform("frame", 0, frame_ty);
    let palette = b.storage("palette", 1, u, false);
    b.vertex("vs", &[BuiltinInput::VertexIndex], |f| {
        let i = vertex_index(f, 0);
        let (one_u, four, one_i) = (f.u32(1), f.i32(4), f.i32(1));
        let axis = |f: &mut Body<'_>, bits: ValueId| {
            let si = f.convert(i32t, bits, Scalar::I32);
            let times = f.binary(BinOp::Mul, si, four);
            let less = f.binary(BinOp::Sub, times, one_i);
            f.convert(fl, less, Scalar::F32)
        };
        let xb = f.binary(BinOp::BitAnd, i, one_u);
        let x = axis(f, xb);
        let yb = f.binary(BinOp::Shr, i, one_u);
        let y = axis(f, yb);
        let (zero, one) = (f.f32(0.0), f.f32(1.0));
        let p = f.construct(v4, &[x, y, zero, one]);
        f.ret_clip(p);
    });
    let colour = b.function("colour", &[u], Some(v3), |f| {
        let i = f.param(0);
        let packed = f.load(u, Builder::element(palette, i));
        let c = f.unpack4x8unorm(packed);
        let rgb = f.let_(v3, Expr::Swizzle(c, vec![0, 1, 2]));
        f.ret(Some(rgb));
    });
    b.fragment("fs", &[BuiltinInput::Position], |f| {
        let field =
            |f: &mut Body<'_>, ty, k| f.load(ty, Builder::whole(frame).with(Proj::Field(k)));
        let splat = |f: &mut Body<'_>, x| f.let_(v3, Expr::Splat(x, 3));
        let fc = f.types().intern(wrela_ir::TypeDef::Struct {
            name: "FragCoord".into(),
            fields: vec![("position".into(), v4)],
        });
        let pos = f.input(0, fc);
        let pos = f.extract(v4, pos, 0);
        let p = f.let_(v2, Expr::Swizzle(pos, vec![0, 1]));
        let colour_of = |f: &mut Body<'_>, k: u32| {
            let at = f.u32(k);
            f.call(colour, &[at]).expect("a colour")
        };
        let (c0, c2) = (colour_of(f, 0), colour_of(f, 2));
        let resolution = field(f, v2, 0);
        let py = f.extract(fl, p, 1);
        let ry = f.extract(fl, resolution, 1);
        let t = f.binary(BinOp::Div, py, ry);
        let ts = splat(f, t);
        let c = f.builtin(v3, B::Mix, &[c0, c2, ts]);
        let center = field(f, v2, 1);
        let radius = field(f, fl, 3);
        let dist = f.builtin(fl, B::Distance, &[p, center]);
        let d = f.binary(BinOp::Sub, dist, radius);
        let c7 = colour_of(f, 7);
        let k = f.f32(0.4);
        let ks = splat(f, k);
        let glow = f.binary(BinOp::Mul, c7, ks);
        let zero = f.f32(0.0);
        let outside = f.builtin(fl, B::Max, &[d, zero]);
        let neg = f.let_(fl, Expr::Unary(wrela_ir::UnOp::Neg, outside));
        let k3 = f.f32(0.3);
        let soft = f.binary(BinOp::Mul, k3, radius);
        let x = f.binary(BinOp::Div, neg, soft);
        let fall = f.builtin(fl, B::Exp, &[x]);
        let falls = splat(f, fall);
        let lit = f.binary(BinOp::Mul, glow, falls);
        let c = f.binary(BinOp::Add, c, lit);
        let time = field(f, fl, 2);
        let three = f.f32(3.0);
        let tt = f.binary(BinOp::Mul, time, three);
        let wave = f.builtin(fl, B::Cos, &[tt]);
        let (k85, k15) = (f.f32(0.85), f.f32(0.15));
        let w = f.binary(BinOp::Mul, k15, wave);
        let pulse = f.binary(BinOp::Add, k85, w);
        let c5 = colour_of(f, 5);
        let pulses = splat(f, pulse);
        let dim = f.binary(BinOp::Mul, c5, pulses);
        let (lo, hi) = (f.f32(-1.0), f.f32(1.0));
        let edge = f.builtin(fl, B::Smoothstep, &[lo, hi, d]);
        let edges = splat(f, edge);
        let c = f.builtin(v3, B::Mix, &[dim, c, edges]);
        let ones = splat(f, hi);
        let c = f.builtin(v3, B::Min, &[c, ones]);
        let (r, g, bl) = (f.extract(fl, c, 0), f.extract(fl, c, 1), f.extract(fl, c, 2));
        let out = f.construct(v4, &[r, g, bl, hi]);
        f.ret(Some(out));
    });
    b
}
