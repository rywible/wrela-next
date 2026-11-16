//! `wrela-host --pipelines` ([`crate::Options::report_pipelines`]): a line on stderr for each
//! pipeline as it's built, saying what it asks of the GPU. A compute pipeline's names the
//! workgroup size its shader declares and, on Metal, what Metal compiled it to: its SIMD width
//! (`threadExecutionWidth`), the most threads a threadgroup of it may have
//! (`maxTotalThreadsPerThreadgroup`) and its static threadgroup memory
//! (`staticThreadgroupMemoryLength`) (Apple's `MTLComputePipelineState`, docs/citations.md).
//!
//! The most threads comes twice. The MSL writer declares each kernel's workgroup as its cap
//! (`[[max_total_threads_per_threadgroup(N)]]`), so as built it's the workgroup's size, unless
//! the kernel's registers hold it lower (then its dispatches fail). Compiled again without the
//! cap, it's as many as the kernel's registers allow, up to the device's: on GPUs whose
//! registers are allotted when a kernel is compiled, below the device's means the registers
//! limit how many of its threads a core holds. On an Apple M4 it's the device's 1024 whatever
//! the kernel (measured on Metal kernels with 1 to 256 live `float4`s); its registers are
//! allotted as it runs (Dynamic Caching, docs/citations.md), so this number can't show their
//! pressure there.
//!
//! wgpu 30 has no way to reach a pipeline's Metal object (`as_hal` reaches the device, not a
//! pipeline, and wgpu-hal keeps its pipeline state private), so on Metal each compute pipeline's
//! MSL, as the build ships it, is compiled again on the raw device (`Device::as_hal`) as wgpu-hal
//! 30 compiles a passthrough shader (`metal/device.rs`: default compile options, no buffer marked
//! immutable). Only a run with the flag pays for it.
//!
//! The one `unsafe` is `as_hal`, which only reads the device.
#![allow(unsafe_code)]

use std::fmt::Write as _;

/// What Metal says of a compute pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Metal {
    /// `threadExecutionWidth`: the threads that run in lockstep (a SIMD group).
    pub simd: u64,
    /// `maxTotalThreadsPerThreadgroup`: the most threads one of its threadgroups may have, as
    /// built: at most its workgroup's size, which the MSL writer declares as the kernel's cap.
    pub max_threads: u64,
    /// `maxTotalThreadsPerThreadgroup` of the kernel compiled without that cap: as many as its
    /// registers allow, up to the device's. Below the device's, the kernel's registers are
    /// what limits how many of its threads a GPU core holds at once.
    pub uncapped: u64,
    /// `staticThreadgroupMemoryLength`: bytes of threadgroup memory its shader declares.
    pub threadgroup_memory: u64,
}

/// What the report says of a pipeline.
pub(crate) enum Report<'a> {
    Compute {
        workgroup: [u32; 3],
        /// Metal's, or why there's none.
        metal: Result<Metal, String>,
        /// The device's most threads in a workgroup.
        device_max: u32,
    },
    Render {
        vertex: &'a str,
        fragment: &'a str,
        /// The variants built for the targets it's drawn into.
        variants: usize,
    },
    /// Not built: the device lacks a feature its shader needs.
    Unavailable { features: &'a [String] },
}

/// The report's line for the pipeline in `shader` (`pipeline_3.metal`) named `name` (its entry
/// points').
pub(crate) fn line(shader: &str, name: &str, report: &Report) -> String {
    let mut s = format!("{shader}  {name}");
    match report {
        Report::Compute { workgroup: [x, y, z], metal, device_max } => {
            let threads = u64::from(*x) * u64::from(*y) * u64::from(*z);
            let _ = write!(s, "  compute  workgroup {x}x{y}x{z} = {threads}");
            match metal {
                Ok(m) => {
                    let _ = write!(
                        s,
                        "  simd {}  max threads {}, uncapped {} of {device_max}  threadgroup memory {} bytes",
                        m.simd, m.max_threads, m.uncapped, m.threadgroup_memory
                    );
                    if threads > m.max_threads {
                        s.push_str("  (its workgroup is over its max)");
                    }
                }
                Err(why) => {
                    let _ = write!(s, "  ({why})");
                }
            }
        }
        Report::Render { vertex, fragment, variants } => {
            let _ = write!(
                s,
                "  render  vertex {vertex}  fragment {fragment}  {variants} target variant{}",
                if *variants == 1 { "" } else { "s" }
            );
        }
        Report::Unavailable { features } => {
            let _ = write!(s, "  not built: the device lacks {}", features.join(", "));
        }
    }
    s
}

/// Metal's account of the compute pipeline of the MSL `source`'s kernel `entry`; an error
/// saying why there's none, on another backend among them.
pub(crate) fn metal(device: &wgpu::Device, source: &str, entry: &str) -> Result<Metal, String> {
    #[cfg(target_vendor = "apple")]
    {
        apple::measure(device, source, entry)
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, source, entry);
        Err("only Metal reports more".into())
    }
}

#[cfg(target_vendor = "apple")]
mod apple {
    use super::Metal;
    use objc2::rc::autoreleasepool;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLCompileOptions, MTLComputePipelineDescriptor, MTLComputePipelineState, MTLDevice,
        MTLLibrary, MTLPipelineOption,
    };

    pub(super) fn measure(device: &wgpu::Device, msl: &str, entry: &str) -> Result<Metal, String> {
        // SAFETY: the raw device is only asked to compile a library and a pipeline of its own;
        // nothing wgpu made is touched or destroyed.
        let raw = match unsafe { device.as_hal::<wgpu::hal::api::Metal>() } {
            Some(hal) => hal.raw_device().clone(),
            None => return Err("only Metal reports more".into()),
        };
        // The MSL writer caps the kernel at its workgroup (`[[max_total_threads_per_threadgroup(N)]]`), so
        // as built its max is the workgroup's size unless registers hold it below that. Without
        // the cap, Metal says how many threads its registers allow.
        let cap = msl
            .find("[[max_total_threads_per_threadgroup(")
            .and_then(|at| Some(at..at + msl[at..].find(")]] ")? + ")]] ".len()));
        let built = compile(&raw, msl, entry)?;
        let uncapped = match cap {
            Some(cap) => {
                let mut text = msl.to_string();
                text.replace_range(cap, "");
                compile(&raw, &text, entry)?.max_threads
            }
            None => built.max_threads,
        };
        Ok(Metal { uncapped, ..built })
    }

    /// `msl`'s kernel `name` compiled on `raw` as wgpu-hal compiles a passthrough shader: what
    /// Metal says of it, its uncapped max not yet known.
    fn compile(
        raw: &ProtocolObject<dyn MTLDevice>,
        msl: &str,
        name: &str,
    ) -> Result<Metal, String> {
        autoreleasepool(|_| {
            let options = MTLCompileOptions::new();
            let library = raw
                .newLibraryWithSource_options_error(&NSString::from_str(msl), Some(&options))
                .map_err(|e| format!("Metal: {e}"))?;
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| format!("Metal: no function {name}"))?;
            let descriptor = MTLComputePipelineDescriptor::new();
            descriptor.setComputeFunction(Some(&function));
            let state = raw
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &descriptor,
                    MTLPipelineOption::empty(),
                    None,
                )
                .map_err(|e| format!("Metal: {e}"))?;
            let max_threads = state.maxTotalThreadsPerThreadgroup() as u64;
            Ok(Metal {
                simd: state.threadExecutionWidth() as u64,
                max_threads,
                uncapped: max_threads,
                threadgroup_memory: state.staticThreadgroupMemoryLength() as u64,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_line_names_the_pipeline_and_what_metal_says() {
        let metal = Metal { simd: 32, max_threads: 64, uncapped: 576, threadgroup_memory: 4096 };
        let compute = Report::Compute { workgroup: [8, 8, 1], metal: Ok(metal), device_max: 1024 };
        assert_eq!(
            line("pipeline_67.metal", "walk_direct", &compute),
            "pipeline_67.metal  walk_direct  compute  workgroup 8x8x1 = 64  simd 32  max threads 64, uncapped 576 of 1024  threadgroup memory 4096 bytes"
        );
        let over = Report::Compute {
            workgroup: [1024, 1, 1],
            metal: Ok(Metal { max_threads: 896, uncapped: 896, ..metal }),
            device_max: 1024,
        };
        assert!(
            line("pipeline_0.metal", "k", &over).ends_with("  (its workgroup is over its max)")
        );
        let elsewhere = Report::Compute {
            workgroup: [64, 1, 1],
            metal: Err("only Metal reports more".into()),
            device_max: 1024,
        };
        assert_eq!(
            line("pipeline_1.metal", "k", &elsewhere),
            "pipeline_1.metal  k  compute  workgroup 64x1x1 = 64  (only Metal reports more)"
        );
        let render = Report::Render { vertex: "cover", fragment: "shade", variants: 1 };
        assert_eq!(
            line("pipeline_2.metal", "cover+shade", &render),
            "pipeline_2.metal  cover+shade  render  vertex cover  fragment shade  1 target variant"
        );
    }
}
