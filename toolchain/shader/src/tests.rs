//! The writers, each on small modules built as IR ([`crate::build`]): what their text says, and
//! that each target's own offline tool takes it (where this machine has the tool).

use crate::build::{BinOp, Builder, BuiltinInput, Scalar};
use crate::{Checks, emit_shader, offline};
use wrela_abi::Target;
use wrela_abi::manifest::{
    BindingKind, BindingStage, Output, Pipeline, RenderTarget, ResourceBinding, Stage,
    UniformBlock, UniformSpace,
};
use wrela_ir as ir;

fn compute_pipeline(bindings: Vec<ResourceBinding>, uniform: Option<UniformBlock>) -> Pipeline {
    Pipeline {
        name: "k".into(),
        shader: "k.metal".into(),
        stage: Stage::Compute { entry: "k".into(), workgroup_size: [64, 1, 1] },
        uniform,
        bindings,
        debug_flag: None,
        requires: Vec::new(),
        conditional: false,
        standin: None,
        sizes: None,
        kept: 0,
        work: Vec::new(),
        unbounded: Vec::new(),
    }
}

fn render_pipeline(bindings: Vec<ResourceBinding>) -> Pipeline {
    Pipeline {
        name: "v+f".into(),
        shader: "vf.metal".into(),
        stage: Stage::Render {
            vertex_entry: "v".into(),
            fragment_entry: "f".into(),
            blends: Vec::new(),
            cull: Default::default(),
            depth_bias: Default::default(),
            depth: Default::default(),
            writes_depth: false,
            outputs: vec![Output::Float],
            targets: vec![RenderTarget::screen(true)],
        },
        uniform: None,
        bindings,
        debug_flag: None,
        requires: Vec::new(),
        conditional: false,
        standin: None,
        sizes: None,
        kept: 0,
        work: Vec::new(),
        unbounded: Vec::new(),
    }
}

fn rw(binding: u32) -> ResourceBinding {
    ResourceBinding {
        binding,
        kind: BindingKind::ReadWrite,
        stage: BindingStage::Both,
        format: None,
    }
}

fn read(binding: u32) -> ResourceBinding {
    ResourceBinding { binding, kind: BindingKind::Read, stage: BindingStage::Both, format: None }
}

/// Both targets' shaders of `m` for `p`, each through its offline tool where this machine has it.
fn written(
    m: &ir::Module,
    p: &Pipeline,
    checks: Checks,
) -> (String, Vec<u32>, Vec<String>, Vec<String>) {
    let mut pm = p.clone();
    let metal = emit_shader(m, &mut pm, Target::Metal, checks).expect("MSL");
    offline::check(offline::Tool::Metal, &metal.bytes).unwrap_or_else(|e| {
        panic!("Apple's compiler rejects:\n{e}\n{}", String::from_utf8_lossy(&metal.bytes))
    });
    let mut pv = p.clone();
    let spv = emit_shader(m, &mut pv, Target::Vulkan, checks).expect("SPIR-V");
    offline::check(offline::Tool::SpirvVal, &spv.bytes)
        .unwrap_or_else(|e| panic!("spirv-val rejects:\n{e}"));
    let words = spv.bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    (String::from_utf8(metal.bytes).expect("UTF-8"), words, metal.entry_points, spv.entry_points)
}

/// The opcodes of a SPIR-V module's instructions, with their operands.
fn instructions(words: &[u32]) -> Vec<(u32, Vec<u32>)> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let (count, op) = ((words[i] >> 16) as usize, words[i] & 0xffff);
        out.push((op, words[i + 1..i + count.max(1)].to_vec()));
        i += count.max(1);
    }
    out
}

fn has_op(words: &[u32], op: spirv::Op) -> bool {
    instructions(words).iter().any(|(o, _)| *o == op as u32)
}

/// A kernel adding 1 to `out[id.x]` for `id.x` below a uniform count, through a loop.
fn kernel(name: &str) -> (ir::Module, Pipeline) {
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let u3 = b.vector(Scalar::U32, 3);
    let params = b.structure("Params", &[("count", u), ("step", u)]);
    let p = b.uniform("params", 0, params);
    let out = b.storage("out", 1, u, true);
    b.compute(name, [64, 1, 1], &[BuiltinInput::GlobalInvocationId], |f| {
        let id = f.input(0, u3);
        let x = f.extract(u, id, 0);
        let count = f.load(u, Builder::whole(p).with(ir::Proj::Field(0)));
        let inside = f.binary(BinOp::Lt, x, count);
        f.if_(
            inside,
            |f| {
                let old = f.load(u, Builder::element(out, x));
                let step = f.load(u, Builder::whole(p).with(ir::Proj::Field(1)));
                let q = f.binary(BinOp::Div, old, step);
                let new = f.binary(BinOp::Add, q, step);
                f.store(Builder::element(out, x), new);
            },
            |_| {},
        );
        f.ret(None);
    });
    let pipeline = compute_pipeline(
        vec![rw(1)],
        Some(UniformBlock { binding: 0, size: 16, space: UniformSpace::Uniform }),
    );
    (b.m, pipeline)
}

#[test]
fn a_kernel_is_written_for_both_targets() {
    let (m, p) = kernel("step_cells");
    let (msl, spv, msl_names, spv_names) = written(&m, &p, Checks::CHECKED);
    assert!(
        msl.contains("[[max_total_threads_per_threadgroup(64)]] kernel void step_cells_e("),
        "{msl}"
    );
    assert!(msl.contains("#pragma METAL fp math_mode(relaxed)"), "the math mode is stated:\n{msl}");
    // Every name the writer makes has a letter of its own: an entry point's is `_e` after it.
    assert_eq!(msl_names, ["step_cells_e"]);
    assert_eq!(spv_names, ["step_cells"]);
    assert_eq!(spv[0], spirv::MAGIC_NUMBER);
    assert_eq!(spv[1], 0x0001_0600, "SPIR-V 1.6");
    // Integer division is guarded on both targets.
    assert!(msl.contains("naga_div("), "{msl}");
    assert!(has_op(&spv, spirv::Op::Select));
}

#[test]
fn a_trusted_shader_has_no_checks() {
    let (m, p) = kernel("k");
    let (checked, cspv, ..) = written(&m, &p, Checks::CHECKED);
    let (trusted, tspv, ..) = written(&m, &p, Checks::UNCHECKED);
    for check in ["_buffer_sizes.size", "naga_div("] {
        assert!(checked.contains(check), "a build checks: no `{check}`");
        assert!(!trusted.contains(check), "a trusted build doesn't: `{check}`");
    }
    assert!(tspv.len() < cspv.len());
}

#[test]
fn the_text_is_unindented_and_struct_names_are_cut() {
    let mut b = Builder::new();
    let f = b.scalar(Scalar::F32);
    let long = "Then_Then_Then_Placed_With_Transform_Of_A_Very_Long_Name";
    let s = b.structure(long, &[("x", f)]);
    let out = b.storage("out", 0, s, true);
    let u = b.scalar(Scalar::U32);
    b.compute("k", [1, 1, 1], &[], |fb| {
        let zero = fb.u32(0);
        let one = fb.f32(1.0);
        let v = fb.construct(s, &[one]);
        fb.store(Builder::element(out, zero), v);
        let _ = u;
    });
    let (msl, ..) = written(&b.m, &compute_pipeline(vec![rw(0)], None), Checks::CHECKED);
    assert!(msl.lines().all(|l| !l.starts_with([' ', '\t'])), "{msl}");
    assert!(msl.contains("Then_Then_Then_Placed_With_Trans {"), "{msl}");
    assert!(!msl.contains(long));
}

/// A vector of three followed tightly by a member is held packed in MSL, at the IR's offsets.
#[test]
fn a_vec3_followed_by_a_member_is_packed() {
    let mut b = Builder::new();
    let f = b.scalar(Scalar::F32);
    let v3 = b.vector(Scalar::F32, 3);
    let corner = b.structure("Corner", &[("at", v3), ("w", f), ("normal", v3)]);
    let out = b.storage("out", 0, corner, true);
    b.compute("k", [1, 1, 1], &[], |fb| {
        let zero = fb.u32(0);
        let c = fb.load(corner, Builder::element(out, zero));
        let at = fb.extract(v3, c, 0);
        let two = fb.f32(2.0);
        let w = fb.extract(f, c, 1);
        let s = fb.binary(BinOp::Mul, w, two);
        let splat = fb.let_(v3, ir::Expr::Splat(s, 3));
        let moved = fb.binary(BinOp::Add, at, splat);
        fb.store(Builder::element(out, zero).with(ir::Proj::Field(2)), moved);
    });
    let (msl, ..) = written(&b.m, &compute_pipeline(vec![rw(0)], None), Checks::CHECKED);
    assert!(msl.contains("metal::packed_float3 m0_at;"), "{msl}");
    // The last vector ends at the struct's span: not packed, its fourth float the padding.
    assert!(msl.contains("metal::float3 m2_normal;"), "{msl}");
}

/// A table of structs with a packed 3-vector member: the member's constant is written as its own
/// packed type, not converted from a vector (a constructor Metal runs at program start, which its
/// compiler from source refuses where it isn't folded: toolchain/tests/table_vec3).
#[test]
fn a_tables_packed_member_is_written_as_its_packed_type() {
    let mut b = Builder::new();
    let f = b.scalar(Scalar::F32);
    let v3 = b.vector(Scalar::F32, 3);
    let cone = b.structure("Cone", &[("a", v3), ("b", v3), ("r", f), ("s", f)]);
    let pair = b.array(cone, 2);
    let num = |x: f32| ir::ConstValue::Scalar(ir::Const::F32(x));
    let vec = |x: f32| ir::ConstValue::Parts(vec![num(x), num(x + 1.0), num(x + 2.0)]);
    let one =
        |x: f32| ir::ConstValue::Parts(vec![vec(x), vec(x + 3.0), num(x + 6.0), num(x + 7.0)]);
    b.m.data.push(ir::Data {
        name: "cones".into(),
        ty: pair,
        value: ir::ConstValue::Parts(vec![one(0.0), one(8.0)]),
    });
    let out = b.storage("out", 0, v3, true);
    b.compute("k", [1, 1, 1], &[], |fb| {
        let zero = fb.u32(0);
        let place = ir::Place::root(ir::PlaceRoot::Data(ir::DataId(0)))
            .with(ir::Proj::Index(zero))
            .with(ir::Proj::Field(1));
        let v = fb.load(v3, place);
        fb.store(Builder::element(out, zero), v);
    });
    let (msl, ..) = written(&b.m, &compute_pipeline(vec![rw(0)], None), Checks::CHECKED);
    assert!(msl.contains("metal::packed_float3 m1_b;"), "{msl}");
    assert!(
        msl.contains("metal::float3(0.0, 1.0, 2.0), metal::packed_float3(3.0, 4.0, 5.0)"),
        "{msl}"
    );
}

/// A kernel's workgroup memory: declared in its body, zeroed by its first invocation before a
/// barrier; and a workgroup variable in SPIR-V.
#[test]
fn workgroup_memory_is_declared_in_the_kernel_and_zeroed() {
    let mut b = Builder::new();
    let f = b.scalar(Scalar::F32);
    let u = b.scalar(Scalar::U32);
    let row = b.array(f, 64);
    let shared = b.workgroup("row", row);
    let out = b.storage("out", 0, f, true);
    b.compute("k", [64, 1, 1], &[BuiltinInput::LocalInvocationId], |fb| {
        let u3 = fb.types().vector_of(Scalar::U32, 3);
        let id = fb.input(0, u3);
        let i = fb.extract(u, id, 0);
        let x = fb.convert(f, i, Scalar::F32);
        fb.store(Builder::whole(shared).with(ir::Proj::Index(i)), x);
        fb.barrier();
        let last = fb.u32(63);
        let j = fb.binary(BinOp::Sub, last, i);
        let y = fb.load(f, Builder::whole(shared).with(ir::Proj::Index(j)));
        fb.store(Builder::element(out, i), y);
    });
    let (msl, spv, ..) = written(&b.m, &compute_pipeline(vec![rw(0)], None), Checks::CHECKED);
    let body = msl.split("kernel void k_e(").nth(1).expect("the kernel");
    let open = body.find(") {").expect("its body");
    assert!(body[open..].contains("threadgroup A"), "{body}");
    assert!(body.contains("__local_invocation_index == 0u"), "{body}");
    let workgroup = instructions(&spv).iter().any(|(op, ops)| {
        *op == spirv::Op::Variable as u32
            && ops.get(2) == Some(&(spirv::StorageClass::Workgroup as u32))
    });
    assert!(workgroup);
}

/// A fragment shader's promised depth: Metal's `depth(greater)` or `depth(less)`, SPIR-V's
/// `DepthGreater` or `DepthLess`; with no promise, `depth(any)` and neither mode.
#[test]
fn a_promised_depth_is_each_targets_own() {
    for promise in [None, Some(ir::DepthPromise::Greater), Some(ir::DepthPromise::Less)] {
        let mut b = Builder::new();
        b.vertex("v", &[BuiltinInput::VertexIndex], |f| {
            let zero = f.f32(0.0);
            let one = f.f32(1.0);
            let v4 = f.types().vector(4);
            let p = f.construct(v4, &[zero, zero, zero, one]);
            f.ret_clip(p);
        });
        let v4 = b.vector(Scalar::F32, 4);
        let f32t = b.scalar(Scalar::F32);
        let out = b.structure("Out", &[("colour", v4), ("depth", f32t)]);
        let func = b.function("f", &[], Some(out), |f| {
            let half = f.f32(0.5);
            let one = f.f32(1.0);
            let c = f.construct(v4, &[half, half, half, one]);
            let o = f.construct(out, &[c, half]);
            f.ret(Some(o));
        });
        b.m.entry_points.push(ir::EntryPoint {
            name: "f".into(),
            stage: ir::Stage::Fragment {
                varyings: None,
                outputs: vec![ir::FragmentColour {
                    path: vec![0],
                    scalar: Scalar::F32,
                    components: 4,
                }],
                depth: Some(vec![1]),
                promise,
            },
            function: func,
            inputs: Vec::new(),
        });
        let (msl, spv, ..) = written(&b.m, &render_pipeline(Vec::new()), Checks::CHECKED);
        let modes: Vec<u32> = instructions(&spv)
            .iter()
            .filter(|(op, _)| *op == spirv::Op::ExecutionMode as u32)
            .map(|(_, ops)| ops[1])
            .collect();
        let (quals, mode) = match promise {
            None => ("[[depth(any)]]", None),
            Some(ir::DepthPromise::Greater) => {
                ("[[depth(greater)]]", Some(spirv::ExecutionMode::DepthGreater))
            }
            Some(ir::DepthPromise::Less) => {
                ("[[depth(less)]]", Some(spirv::ExecutionMode::DepthLess))
            }
        };
        assert!(msl.contains(quals), "{msl}");
        assert!(modes.contains(&(spirv::ExecutionMode::DepthReplacing as u32)));
        for m in [spirv::ExecutionMode::DepthGreater, spirv::ExecutionMode::DepthLess] {
            assert_eq!(modes.contains(&(m as u32)), Some(m) == mode, "{promise:?}");
        }
    }
}

/// Derivatives are fine on both targets: Metal's own (a pixel's against its neighbour's), and
/// SPIR-V's `Fine` ones (RADV takes the coarse ones otherwise, one a quad).
#[test]
fn derivatives_are_fine() {
    let mut b = Builder::new();
    b.vertex("v", &[], |f| {
        let zero = f.f32(0.0);
        let v4 = f.types().vector(4);
        let p = f.construct(v4, &[zero, zero, zero, zero]);
        f.ret_clip(p);
    });
    b.fragment("f", &[BuiltinInput::Position], |f| {
        let v4 = f.types().vector(4);
        let fc = f.types().intern(ir::TypeDef::Struct {
            name: "FragCoord".into(),
            fields: vec![("p".into(), v4)],
        });
        let coord = f.input(0, fc);
        let p = f.extract(v4, coord, 0);
        let dx = f.builtin(v4, ir::Builtin::Dpdx, &[p]);
        let dy = f.builtin(v4, ir::Builtin::Fwidth, &[p]);
        let s = f.binary(BinOp::Add, dx, dy);
        f.ret(Some(s));
    });
    let (msl, spv, ..) = written(&b.m, &render_pipeline(Vec::new()), Checks::CHECKED);
    assert!(msl.contains("metal::dfdx(") && msl.contains("metal::fwidth("), "{msl}");
    assert!(has_op(&spv, spirv::Op::DPdxFine) && has_op(&spv, spirv::Op::FwidthFine));
    assert!(!has_op(&spv, spirv::Op::DPdx));
}

/// Float constants at the edges of what a float holds are written so each tool reads them back:
/// the largest, the smallest normal, infinities and NaN.
#[test]
fn float_constants_at_the_edges_are_read_back() {
    let mut b = Builder::new();
    let f = b.scalar(Scalar::F32);
    let out = b.storage("out", 0, f, true);
    let values = [
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE,
        1.0e-30,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    b.compute("k", [1, 1, 1], &[], |fb| {
        for (k, x) in values.iter().enumerate() {
            let at = fb.u32(k as u32);
            let v = fb.f32(*x);
            fb.store(Builder::element(out, at), v);
        }
    });
    let (msl, ..) = written(&b.m, &compute_pipeline(vec![rw(0)], None), Checks::CHECKED);
    assert!(msl.contains("INFINITY") && msl.contains("NAN"), "{msl}");
}

/// A sized buffer's length is read from the pipeline's sizes binding on Metal (in its slot,
/// after the rest), and is `OpArrayLength` on Vulkan.
#[test]
fn buffer_sizes_come_from_their_own_binding() {
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let a = b.storage("a", 1, u, false);
    let out = b.storage("out", 2, u, true);
    b.compute("k", [1, 1, 1], &[], |fb| {
        let la = fb.let_(u, ir::Expr::ArrayLength(Builder::whole(a)));
        let lb = fb.let_(u, ir::Expr::ArrayLength(Builder::whole(out)));
        let s = fb.binary(BinOp::Add, la, lb);
        let zero = fb.u32(0);
        fb.store(Builder::element(out, zero), s);
    });
    let p = compute_pipeline(vec![read(1), rw(2)], None);
    assert_eq!(
        crate::metal::buffer_sizes(&b.m, &p),
        Some(wrela_abi::manifest::BufferSizes { binding: 3, buffers: vec![1, 2] })
    );
    let (msl, spv, ..) = written(&b.m, &p, Checks::CHECKED);
    assert!(msl.contains("constant _mslBufferSizes& _buffer_sizes [[buffer(2)]]"), "{msl}");
    assert!(has_op(&spv, spirv::Op::ArrayLength));
}

/// A fragment shader rebuilds the vertex's output struct it takes (nested structs and a matrix's
/// columns) from the varyings that hold its parts.
#[test]
fn a_fragment_rebuilds_nested_varyings() {
    let mut m = ir::Module::default();
    let (v4, mat) = (m.types.vector(4), m.types.intern(ir::TypeDef::Matrix(2, 2)));
    let inner = m.types.intern(ir::TypeDef::Struct {
        name: "Inner".into(),
        fields: vec![("matrix".into(), mat), ("colour".into(), v4)],
    });
    let outer = m.types.intern(ir::TypeDef::Struct {
        name: "Outer".into(),
        fields: vec![("position".into(), v4), ("inner".into(), inner)],
    });
    let map: Vec<ir::Varying> = [vec![1, 1], vec![0], vec![1, 0, 1], vec![1, 0, 0]]
        .into_iter()
        .map(|path| ir::Varying { position: path == [0], path, flat: false })
        .collect();
    // The vertex shader returns an `Outer`; the fragment shader its inner colour.
    let mut vf = ir::Function::new("vertex", Vec::new(), Some(outer));
    let z = vf.new_value(outer);
    vf.body = vec![ir::Stmt::Let(z, ir::Expr::Zero(outer)), ir::Stmt::Return(Some(z))];
    let vertex = m.add_function(vf);
    m.entry_points.push(ir::EntryPoint {
        name: "vertex".into(),
        stage: ir::Stage::Vertex { varyings: map.clone() },
        function: vertex,
        inputs: Vec::new(),
    });
    let mut f = ir::Function::new("fragment", vec![ir::Param::value("input", outer)], Some(v4));
    let (input, nested, colour) = (f.new_value(outer), f.new_value(inner), f.new_value(v4));
    f.body = vec![
        ir::Stmt::Let(input, ir::Expr::Param(0)),
        ir::Stmt::Let(nested, ir::Expr::Extract(input, 1)),
        ir::Stmt::Let(colour, ir::Expr::Extract(nested, 1)),
        ir::Stmt::Return(Some(colour)),
    ];
    let function = m.add_function(f);
    m.entry_points.push(ir::EntryPoint {
        name: "fragment".into(),
        stage: ir::Stage::Fragment {
            varyings: Some((outer, map)),
            outputs: vec![ir::FragmentColour {
                path: Vec::new(),
                scalar: Scalar::F32,
                components: 4,
            }],
            depth: None,
            promise: None,
        },
        function,
        inputs: Vec::new(),
    });
    let mut p = render_pipeline(Vec::new());
    if let Stage::Render { vertex_entry, fragment_entry, .. } = &mut p.stage {
        *vertex_entry = "vertex".into();
        *fragment_entry = "fragment".into();
    }
    let (msl, ..) = written(&m, &p, Checks::CHECKED);
    // The inner struct is rebuilt from its members: the matrix from its two columns.
    assert!(msl.contains("metal::float2x2(varyings.v3, varyings.v2)"), "{msl}");
}

/// A body that ends in a trap (a loop left only by `return`) returns once, at its end.
#[test]
fn a_trap_at_the_end_returns_a_zero() {
    let mut b = Builder::new();
    let f = b.scalar(Scalar::F32);
    let out = b.storage("out", 0, f, true);
    let spin = b.function("spin", &[f], Some(f), |fb| {
        let x = fb.param(0);
        fb.loop_(|fb| fb.ret(Some(x)));
        fb.trap();
    });
    b.compute("k", [1, 1, 1], &[], |fb| {
        let one = fb.f32(1.0);
        let r = fb.call(spin, &[one]).expect("a value");
        let zero = fb.u32(0);
        fb.store(Builder::element(out, zero), r);
    });
    let (msl, ..) = written(&b.m, &compute_pipeline(vec![rw(0)], None), Checks::CHECKED);
    let spin_text = msl.split("f0_spin(").nth(1).expect("spin");
    let spin_text = &spin_text[..spin_text.find("\n}\n\n").expect("its end")];
    assert!(spin_text.trim_end().ends_with("return float {};"), "{spin_text}");
}
