//! Window presentation: the surface and the fullscreen copy pass into each swapchain frame.

use super::*;

/// A window the screen is shown in: its surface, and the pass that draws the screen into each
/// of its frames (a triangle over the frame, each pixel the screen's texel there).
pub(super) struct Presenter {
    /// The window, kept to make its surface again on a new device (`Gpu::recover`).
    pub(super) window: Arc<dyn wrela_gpu::Window>,
    /// What its surface's present ids are offset by, and the next id the host gives: a surface
    /// made again counts its ids from 0, and the window's pacing reads them as only rising.
    pub(super) id_base: u64,
    pub(super) next_id: u64,
    pub(super) surface: wrela_gpu::Surface,
    /// The format its pipeline draws: the surface's.
    pub(super) format: TextureFormat,
    pub(super) config: wrela_gpu::SurfaceConfig,
    pub(super) pipeline: wrela_gpu::RenderPipeline,
    pub(super) layout: wrela_gpu::BindGroupLayout,
}

/// The present pass's pipeline for frames of `format`, and its layout (the screen at binding 0).
pub(super) fn present_pipeline(
    device: &wrela_gpu::Device,
    format: TextureFormat,
) -> Result<(wrela_gpu::RenderPipeline, wrela_gpu::BindGroupLayout)> {
    use wrela_abi::manifest::{BindingKind, BindingStage, Output, RenderTarget, ResourceBinding};
    let gpu_err = |e: wrela_gpu::Error| Error::Gpu(e.0);
    let mut entry = wrela_abi::manifest::Pipeline {
        name: "present".into(),
        shader: "present".into(),
        stage: Stage::Render {
            vertex_entry: "cover".into(),
            fragment_entry: "show".into(),
            blends: Vec::new(),
            cull: Cull::None,
            depth_bias: DepthBias::default(),
            depth: DepthState::default(),
            writes_depth: false,
            outputs: vec![Output::Float],
            targets: vec![RenderTarget::of(&[format], false)],
        },
        uniform: None,
        bindings: vec![ResourceBinding {
            binding: 0,
            kind: BindingKind::Texture,
            stage: BindingStage::Fragment,
            format: Some(TextureFormat::Rgba8),
        }],
        debug_flag: None,
        requires: Vec::new(),
        conditional: false,
        standin: None,
        sizes: None,
        kept: 0,
        work: vec![Some(16), Some(16)],
        unbounded: Vec::new(),
    };
    let shader = present_shader()
        .write(&mut entry, wrela_abi::Target::HERE)
        .map_err(|e| Error::Gpu(format!("the present pass's shader: {e}")))?;
    let entries: Vec<wrela_gpu::LayoutEntry> = wrela_abi::metal::layout(&entry)
        .iter()
        .map(|e| wrela_gpu::LayoutEntry {
            binding: e.binding,
            visibility: e.visibility,
            ty: binding_type(e, &entry),
            index_at: if e.table {
                wrela_abi::table::index_offset(&entry, e.binding)
            } else {
                None
            },
        })
        .collect();
    let layout = device.create_bind_group_layout(&entries);
    let module = device.create_shader_module("present", &shader.bytes).map_err(gpu_err)?;
    let Stage::Render { vertex_entry, fragment_entry, .. } = &entry.stage else {
        unreachable!("a render pipeline");
    };
    let pipeline = device
        .create_render_pipeline(&wrela_gpu::RenderPipelineDesc {
            label: "present",
            layout: &layout,
            module: &module,
            vertex_entry,
            fragment_entry,
            colors: &[wrela_gpu::ColorTarget { format, blend: Blend::Replace }],
            depth: None,
            cull: Cull::None,
        })
        .map_err(gpu_err)?;
    Ok((pipeline, layout))
}

/// The present pass: `src` (the screen) drawn into `dst`, a frame of the pipeline's format.
pub(super) fn draw_presented(
    device: &wrela_gpu::Device,
    pipeline: &wrela_gpu::RenderPipeline,
    layout: &wrela_gpu::BindGroupLayout,
    src: &wrela_gpu::TextureView,
    dst: &wrela_gpu::TextureView,
) -> wrela_gpu::CommandBuffer {
    let group = device.create_bind_group(
        layout,
        &[wrela_gpu::BindGroupEntry { binding: 0, resource: wrela_gpu::Resource::View(src) }],
    );
    let mut encoder = device.create_command_encoder("present");
    {
        let colors = [wrela_gpu::ColorAttachment {
            view: dst,
            load: wrela_gpu::LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
        }];
        let mut rp = encoder.begin_render_pass(&wrela_gpu::RenderPassDesc {
            label: "present",
            colors: &colors,
            depth: None,
            timestamps: None,
        });
        rp.set_pipeline(pipeline);
        rp.set_bind_group(&group, &[]);
        rp.draw(0..3, 0..1);
    }
    encoder.finish()
}

/// The present pass's shader, as IR:
/// ```text
/// @group(0) @binding(0) var screen: texture_2d<f32>;
/// @vertex fn cover(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
///     return vec4f(f32(i & 1) * 4 - 1, f32(i >> 1) * 4 - 1, 0, 1);
/// }
/// @fragment fn show(@builtin(position) p: vec4f) -> @location(0) vec4f {
///     return textureLoad(screen, vec2u(p.xy), 0);
/// }
/// ```
fn present_shader() -> wrela_shader::build::Builder {
    use wrela_shader::build::{
        BinOp, Builder, BuiltinInput, Scalar, TexDim, TextureFormat as IrFormat,
    };
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let fl = b.scalar(Scalar::F32);
    let v4 = b.vector(Scalar::F32, 4);
    let screen = b.texture("screen", 0, TexDim::D2, IrFormat::Rgba8, false);
    b.vertex("cover", &[BuiltinInput::VertexIndex], |f| {
        let i = f.input(0, u);
        let one = f.u32(1);
        let xb = f.binary(BinOp::BitAnd, i, one);
        let yb = f.binary(BinOp::Shr, i, one);
        let (four, minus, zero, unit) = (f.f32(4.0), f.f32(1.0), f.f32(0.0), f.f32(1.0));
        let corner = |f: &mut wrela_shader::build::Body<'_>, bit| {
            let v = f.convert(fl, bit, Scalar::F32);
            let v = f.binary(BinOp::Mul, v, four);
            f.binary(BinOp::Sub, v, minus)
        };
        let (x, y) = (corner(f, xb), corner(f, yb));
        let p = f.construct(v4, &[x, y, zero, unit]);
        f.ret_clip(p);
    });
    b.fragment("show", &[BuiltinInput::Position], |f| {
        let p = f.input(0, v4);
        let (x, y) = (f.extract(fl, p, 0), f.extract(fl, p, 1));
        let (x, y) = (f.convert(u, x, Scalar::U32), f.convert(u, y, Scalar::U32));
        let level = f.u32(0);
        let texel = f.texel(screen, x, y, level);
        f.ret(Some(texel));
    });
    b
}
