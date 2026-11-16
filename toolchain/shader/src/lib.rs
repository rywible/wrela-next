//! The GPU back end: a GPU module of the IR ([`wrela_ir`]) written as its target takes it, by the
//! compiler's own writers: Metal shading language for Metal ([`metal`]), SPIR-V for Vulkan
//! ([`spirv`]), each binding where the host's layout puts it ([`wrela_abi::metal`],
//! [`wrela_abi::vulkan`]). Both compute what naga 30's writers computed for the same module
//! (the compiler wrote its shaders through naga until toolchain 0.33's backend change; its
//! writers were read as reference, docs/citations.md), and go beyond them where a target can say
//! more (Metal's conservative depth, its math mode stated) or its rules have moved (Vulkan's
//! explicit layout only in buffers).
//!
//! - Types keep the IR's layout (`wrela_ir::layout`), so a struct member's offset on the GPU is
//!   the offset the CPU wrote uniforms with. An enum is the one exception: the GPU holds it as a
//!   struct of its tag and every payload, and types that hold one are laid out around that
//!   struct. An enum is never `GpuData`, so those types never cross to the CPU.
//! - Evaluation order is the IR's: a value that reads memory, calls or has an effect is computed
//!   where the IR defines it.
//! - GPU modules arrive flattened (`wrela_ir::opt`): every call is inlined but those to the big
//!   functions that take only values, which are written as functions, callees first.
//!
//! A shader the compiler doesn't compile from wrela (the host's own kernels, a host test's
//! shaders) is built as IR in Rust ([`build`]) and written the same way. Every shader the test
//! suite builds goes through its platform's offline tool ([`offline`]).

pub mod build;
pub mod metal;
pub mod offline;
pub mod spirv;

use wrela_ir as ir;

type R<T> = Result<T, String>;

/// The run-time checks a shader is written with. [`Checks::CHECKED`] is what the language's
/// robust indexing and defined integer division rest on (language.md §11). [`Checks::UNCHECKED`]
/// is a trusted build's (`wrela build --trusted-shaders`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Checks {
    /// Indexes clamped into range, else unchecked.
    pub bounds: bool,
    /// Every loop given a bound the GPU's compiler can see, so it can't assume the loop ends.
    pub loops: bool,
    /// `/` and `%` of integers guarded against 0 and the most negative divided by -1.
    pub int_div: bool,
}

impl Checks {
    pub const CHECKED: Checks = Checks { bounds: true, loops: true, int_div: true };
    pub const UNCHECKED: Checks = Checks { bounds: false, loops: false, int_div: false };
}

/// A pipeline's shader as its target takes it: Metal shading language's text, or SPIR-V's words
/// (little-endian); and its entry points' names in it, in the module's order.
#[derive(Clone, Debug)]
pub struct Shader {
    pub bytes: Vec<u8>,
    pub entry_points: Vec<String>,
}

/// A shader file's extension on `target`.
pub fn extension(target: wrela_abi::Target) -> &'static str {
    match target {
        wrela_abi::Target::Metal => "metal",
        wrela_abi::Target::Vulkan => "spv",
    }
}

/// Writes a GPU module for pipeline `p` as `target` takes it (Metal: [`metal::write`]; Vulkan:
/// [`spirv::write`]), with `checks` written in, setting `p.sizes` to the binding Metal reads its
/// buffers' sizes from.
pub fn emit_shader(
    m: &ir::Module,
    p: &mut wrela_abi::manifest::Pipeline,
    target: wrela_abi::Target,
    checks: Checks,
) -> R<Shader> {
    // The fragment shader's promise of its depth: conservative depth.
    let promise = m.entry_points.iter().find_map(|e| match &e.stage {
        ir::Stage::Fragment { depth: Some(_), promise: Some(d), .. } => Some(*d),
        _ => None,
    });
    let out = match target {
        wrela_abi::Target::Metal => {
            p.sizes = metal::buffer_sizes(m, p);
            let depth = promise.map(|d| match d {
                ir::DepthPromise::Greater => metal::ConservativeDepth::Greater,
                ir::DepthPromise::Less => metal::ConservativeDepth::Less,
            });
            let out = metal::write(m, p, depth, checks)?;
            Shader { bytes: out.text.into_bytes(), entry_points: out.entry_points }
        }
        wrela_abi::Target::Vulkan => {
            p.sizes = None;
            let (bytes, entry_points) = spirv::write(m, p, promise, checks)?;
            Shader { bytes, entry_points }
        }
    };
    if out.entry_points.len() != m.entry_points.len() {
        return Err("internal: the shader writer lost an entry point".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
