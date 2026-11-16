//! The Metal back end: the naga module [`crate::to_naga`] builds, written as Metal shading
//! language (MSL) by naga's own MSL writer, with each binding in the argument-table slot
//! [`wrela_abi::metal`] gives it, for the host to hand the GPU as it is.
//!
//! naga's writer is what wgpu ran on the WGSL before (the same options as wgpu 30's Metal backend
//! and its default runtime checks: bounds clamped, loops bounded, integer division guarded,
//! workgroup memory zeroed), so a pipeline computes what it computed then. Two things wgpu did
//! for its own translations it doesn't do for a passthrough shader, and they are done here:
//!
//! - **Buffer sizes.** naga reads a runtime array's length from a `_mslBufferSizes` argument,
//!   which wgpu fills only for its own translations. Here it is a uniform buffer of the
//!   pipeline's own ([`wrela_abi::manifest::BufferSizes`]), which the host fills.
//! - **Threadgroup memory.** naga makes each workgroup variable a kernel argument
//!   (`threadgroup T& name`), whose length wgpu sets only for its own translations. Here each
//!   becomes a declaration in the kernel's body, so Metal sizes it from its type ([`patch`]).
//!
//! naga's writer can't say a fragment's depth is never nearer than its triangle's (Metal's
//! conservative depth); [`patch`] adds it to the text. Both patches are text edits of naga's
//! output, in one place, tested on its exact shapes: a hand-written MSL emitter replaces them.

use naga::back::msl;
use wrela_abi::manifest::{BufferSizes, Pipeline};
use wrela_abi::metal::{self as slots, ShaderStage, Table};

type R<T> = Result<T, String>;

/// The MSL language version written: what wgpu 30 compiles with on macOS 26 (`metal/adapter.rs`),
/// and what Metal compiles a passthrough shader with there (its default, the newest).
pub const LANGUAGE: (u8, u8) = (4, 0);

/// A GPU module written as MSL.
#[derive(Clone, Debug)]
pub struct Msl {
    pub text: String,
    /// Each entry point's name in `text`, in the module's order (naga's namer renames any that
    /// would clash with MSL's own).
    pub entry_points: Vec<String>,
}

pub use crate::Checks;

/// What a fragment shader's depth promises (`std::gpu::WithDepthBehind`): never nearer than its
/// triangle's depth, so the GPU may test the triangle's depth before the shader runs. Which way
/// "nearer" is depends on the pipeline's depth test: `Greater` for a test that keeps less
/// (`less`, `less-equal`: the plain convention), `Less` for one that keeps greater (reversed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConservativeDepth {
    Greater,
    Less,
}

impl ConservativeDepth {
    /// Metal's qualifier.
    fn qualifier(self) -> &'static str {
        match self {
            ConservativeDepth::Greater => "[[depth(greater)]]",
            ConservativeDepth::Less => "[[depth(less)]]",
        }
    }
}

/// The bindings of the module's globals that hold a runtime-sized array, in its order: what
/// naga's `_mslBufferSizes` holds a size for, each a member.
pub fn sized_bindings(module: &naga::Module) -> Vec<u32> {
    module
        .global_variables
        .iter()
        .filter(|(_, g)| module.types[g.ty].inner.needs_host_buffer_byte_size(&module.types))
        .filter_map(|(_, g)| g.binding.as_ref().map(|b| b.binding))
        .collect()
}

/// The pipeline's sizes binding, if its shader reads sizes: one past every other binding.
pub fn buffer_sizes(module: &naga::Module, p: &Pipeline) -> Option<BufferSizes> {
    let buffers = sized_bindings(module);
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

/// `module` as MSL for pipeline `p`, whose manifest entry (bindings, uniform block, debug flag
/// and `sizes`, from [`buffer_sizes`]) says where its bindings go; with `depth` on its fragment
/// shader's depth, and `checks` written in.
pub fn write(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    p: &Pipeline,
    depth: Option<ConservativeDepth>,
    checks: Checks,
) -> R<Msl> {
    let layout = slots::layout(p);
    let mut map = msl::EntryPointResourceMap::default();
    for ep in &module.entry_points {
        let stage = match ep.stage {
            naga::ShaderStage::Vertex => ShaderStage::Vertex,
            naga::ShaderStage::Fragment => ShaderStage::Fragment,
            naga::ShaderStage::Compute => ShaderStage::Compute,
            s => return Err(format!("internal: a {s:?} entry point")),
        };
        let st = slots::slots(&layout, stage);
        let slots_max = wrela_abi::TargetLimits::METAL.buffer_slots_per_stage.unwrap_or(31);
        if st.buffers() > slots_max {
            return Err(format!(
                "internal: `{}` binds {} buffers, past Metal's {}",
                ep.name,
                st.buffers(),
                slots_max
            ));
        }
        let mut resources = msl::BindingMap::default();
        for s in &st.slots {
            let mut target = msl::BindTarget { mutable: s.mutable, ..Default::default() };
            match s.table {
                Table::Buffer => target.buffer = Some(s.index),
                Table::Texture => target.texture = Some(s.index),
                Table::Sampler => {
                    target.sampler = Some(msl::BindSamplerTarget::Resource(s.index));
                }
            }
            resources.insert(naga::ResourceBinding { group: 0, binding: s.binding }, target);
        }
        let sizes_buffer = p.sizes.as_ref().and_then(|s| st.of(s.binding)).map(|s| s.index);
        map.insert(
            ep.name.clone(),
            msl::EntryPointResources { resources, immediates_buffer: None, sizes_buffer },
        );
    }
    let policy = if checks.bounds {
        naga::proc::BoundsCheckPolicy::Restrict
    } else {
        naga::proc::BoundsCheckPolicy::Unchecked
    };
    let options = msl::Options {
        lang_version: LANGUAGE,
        per_entry_point_map: map,
        inline_samplers: Vec::new(),
        spirv_cross_compatibility: false,
        fake_missing_bindings: false,
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: policy,
            buffer: policy,
            image_load: policy,
            binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
        },
        zero_initialize_workgroup_memory: true,
        force_loop_bounding: checks.loops,
        task_dispatch_limits: None,
        mesh_shader_primitive_indices_clamp: true,
        emit_int_div_checks: checks.int_div,
        ray_query_initialization_tracking: true,
    };
    let pipeline_options = msl::PipelineOptions {
        entry_point: None,
        allow_and_force_point_size: false,
        // No vertex buffers: vertices are pulled from storage by the shader itself.
        vertex_pulling_transform: false,
        vertex_buffer_mappings: Vec::new(),
        binding_array_length_map: Default::default(),
    };
    let (text, translated) = msl::write_string(module, info, &options, &pipeline_options)
        .map_err(|e| format!("internal: writing MSL failed: {e:?}"))?;
    let entry_points = translated
        .entry_point_names
        .into_iter()
        .map(|n| n.map_err(|e| format!("internal: an MSL entry point: {e:?}")))
        .collect::<R<Vec<_>>>()?;
    let text = patch::threadgroup_in_body(&text, module)?;
    let text = match depth {
        Some(d) => patch::conservative_depth(&text, d)?,
        None => text,
    };
    // Without indentation, which C++ ignores, as the WGSL was (`crate::unindented`): a
    // creature's item dispatch nests an `if` for each item, and naga's four spaces a level are
    // a large part of its text.
    Ok(Msl { text: crate::unindented(&text), entry_points })
}

/// The edits made to naga 30's MSL text, each for what its writer can't say. A hand-written MSL
/// emitter (one that writes text from the IR, as `to_naga` builds naga's module) writes these
/// itself and replaces this module.
pub mod patch {
    use super::{ConservativeDepth, R};

    /// The fragment output's depth with the promise `d`: naga writes `[[depth(any)]]`, the one
    /// it knows, on the one fragment output that has a depth.
    pub fn conservative_depth(text: &str, d: ConservativeDepth) -> R<String> {
        const ANY: &str = "[[depth(any)]]";
        match text.matches(ANY).count() {
            1 => Ok(text.replacen(ANY, d.qualifier(), 1)),
            n => Err(format!("internal: the MSL has {n} depth outputs, not one, to promise")),
        }
    }

    /// Each kernel's workgroup memory declared in its body. naga writes each workgroup variable
    /// as an argument of the kernel that uses it (`, threadgroup T& name`, one a line), whose
    /// length wgpu sets only for shaders it translates itself; a declaration in the kernel's body
    /// (`threadgroup T name;`) is sized by Metal from its type. Every use reads the same name,
    /// a reference before and a variable now.
    pub fn threadgroup_in_body(text: &str, module: &naga::Module) -> R<String> {
        let shared = module
            .global_variables
            .iter()
            .filter(|(_, g)| g.space == naga::AddressSpace::WorkGroup)
            .count();
        if shared == 0 {
            return Ok(text.to_string());
        }
        let mut out = String::with_capacity(text.len());
        let mut moved: Vec<String> = Vec::new();
        let mut found = 0;
        let mut in_kernel = false;
        for line in text.split_inclusive('\n') {
            let bare = line.trim_end_matches('\n');
            // `[[max_total_threads_per_threadgroup(N)]] kernel void name(`
            if bare.contains("kernel void ") && bare.ends_with('(') {
                in_kernel = true;
            }
            if in_kernel
                && let Some(rest) = bare.strip_prefix(", threadgroup ")
                && let Some((ty, name)) = rest.rsplit_once("& ")
                && !name.contains([' ', '[', '('])
            {
                moved.push(format!("    threadgroup {ty} {name};\n"));
                found += 1;
                continue;
            }
            out.push_str(line);
            if in_kernel && bare == ") {" {
                in_kernel = false;
                for m in moved.drain(..) {
                    out.push_str(&m);
                }
            }
        }
        if found == 0 || !moved.is_empty() {
            return Err(format!(
                "internal: the MSL's workgroup memory ({shared} variables) isn't in the arguments naga writes"
            ));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(wgsl: &str) -> (naga::Module, naga::valid::ModuleInfo) {
        let m = naga::front::wgsl::parse_str(wgsl).expect("WGSL");
        let info = crate::validate(&m).expect("valid");
        (m, info)
    }

    /// A kernel's workgroup memory is declared in its body, not passed in; the rest of naga's
    /// text is as it was.
    #[test]
    fn workgroup_memory_is_declared_in_the_kernel() {
        let (m, info) = module(
            "var<workgroup> row: array<f32, 64>;
             var<workgroup> total: atomic<u32>;
             @group(0) @binding(0) var<storage, read_write> out: array<f32>;
             @compute @workgroup_size(64) fn k(@builtin(local_invocation_index) i: u32) {
                 row[i] = f32(i);
                 atomicAdd(&total, 1u);
                 workgroupBarrier();
                 out[i] = row[63u - i] + f32(atomicLoad(&total));
             }",
        );
        let mut p = compute_pipeline(&m);
        p.sizes = buffer_sizes(&m, &p);
        let msl = write(&m, &info, &p, None, Checks::CHECKED).expect("MSL");
        assert!(!msl.text.contains(", threadgroup "), "{}", msl.text);
        assert_eq!(msl.text.matches("\nthreadgroup ").count(), 2, "{}", msl.text);
        let body = msl.text.split("kernel void k(").nth(1).expect("the kernel");
        let open = body.find(") {").expect("its body");
        assert!(body[open..].contains("threadgroup type_1 row;"), "{body}");
        assert!(body[open..].contains("threadgroup metal::atomic_uint total;"), "{body}");
    }

    /// A promised depth replaces naga's `depth(any)`; a shader with no depth can't promise one.
    #[test]
    fn a_promised_depth_is_written_as_metals_qualifier() {
        let (m, info) = module(
            "struct Out { @location(0) c: vec4f, @builtin(frag_depth) d: f32 }
             @vertex fn v() -> @builtin(position) vec4f { return vec4f(0.0); }
             @fragment fn f(@builtin(position) p: vec4f) -> Out { return Out(vec4f(1.0), p.z); }",
        );
        let p = render_pipeline();
        let plain = write(&m, &info, &p, None, Checks::CHECKED).expect("MSL");
        assert!(plain.text.contains("[[depth(any)]]"));
        let behind =
            write(&m, &info, &p, Some(ConservativeDepth::Greater), Checks::CHECKED).expect("MSL");
        assert!(behind.text.contains("float d [[depth(greater)]];"), "{}", behind.text);
        assert!(!behind.text.contains("depth(any)"));
        let reversed =
            write(&m, &info, &p, Some(ConservativeDepth::Less), Checks::CHECKED).expect("MSL");
        assert!(reversed.text.contains("[[depth(less)]]"));
        assert!(patch::conservative_depth("no depth", ConservativeDepth::Greater).is_err());
    }

    /// A runtime array's length is read from the pipeline's sizes binding, in its slot.
    #[test]
    fn buffer_sizes_are_read_from_their_own_binding() {
        let (m, info) = module(
            "@group(0) @binding(1) var<storage, read> a: array<u32>;
             @group(0) @binding(2) var<storage, read_write> b: array<u32>;
             @compute @workgroup_size(1) fn k() { b[0] = arrayLength(&a) + arrayLength(&b); }",
        );
        let mut p = compute_pipeline(&m);
        p.sizes = buffer_sizes(&m, &p);
        assert_eq!(p.sizes, Some(BufferSizes { binding: 3, buffers: vec![1, 2] }));
        let msl = write(&m, &info, &p, None, Checks::CHECKED).expect("MSL");
        assert!(
            msl.text.contains("constant _mslBufferSizes& _buffer_sizes [[buffer(2)]]"),
            "{}",
            msl.text
        );
        assert_eq!(msl.entry_points, ["k"]);
        let unchecked = write(&m, &info, &p, None, Checks::UNCHECKED).expect("MSL");
        assert!(unchecked.text.len() < msl.text.len());
    }

    fn compute_pipeline(m: &naga::Module) -> Pipeline {
        use wrela_abi::manifest::{BindingKind, ResourceBinding, Stage};
        let bindings = m
            .global_variables
            .iter()
            .filter_map(|(_, g)| {
                let b = g.binding.as_ref()?.binding;
                let kind = match g.space {
                    naga::AddressSpace::Storage { access }
                        if access.contains(naga::StorageAccess::STORE) =>
                    {
                        BindingKind::ReadWrite
                    }
                    _ => BindingKind::Read,
                };
                Some(ResourceBinding { binding: b, kind, stage: Default::default(), format: None })
            })
            .collect();
        Pipeline {
            name: "k".into(),
            shader: "k.metal".into(),
            stage: Stage::Compute { entry: "k".into(), workgroup_size: [64, 1, 1] },
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

    fn render_pipeline() -> Pipeline {
        use wrela_abi::manifest::{Output, RenderTarget, Stage};
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
                writes_depth: true,
                outputs: vec![Output::Float],
                targets: vec![RenderTarget::screen(true)],
            },
            uniform: None,
            bindings: Vec::new(),
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
}
