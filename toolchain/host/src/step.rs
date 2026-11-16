//! The step's guard: a compute dispatch is refused before it runs when its work could hold the
//! GPU past [`STEP_MS`] (AGENTS.md, rule 4; docs/rates.md, "Bounded compute steps").
//!
//! On 2026-11-15 one dispatch (a reference that marched a medium at every pixel with fine steps)
//! ran so long that the GPU restarted and the machine went down. A dispatch already running can't
//! be stopped, and the display's guard (`Behind`, gpu.rs) only stops the frames after it. So the
//! host weighs each dispatch first. The compiler states, for each kernel, the most work one
//! invocation can do (its work count, language.md §18.11, the manifest's `work`), or that it
//! can't say (a loop with no exit it is sure of). A dispatch's work is that bound times its
//! invocations (its groups times the kernel's workgroup size, and never fewer than the GPU runs
//! side by side: [`width`]), and [`unit_ns`] turns work into time on this machine. A dispatch
//! whose kernel can't say, or whose work could take longer than the limit, is refused with an
//! error that names the kernel, its bound, its invocations and the limit. Nothing turns the guard
//! off: [`STEP_VAR`] may only lower the limit (to show the refusal with a small kernel).
//!
//! An indirect dispatch's groups are in a GPU buffer the host doesn't read before it runs: its
//! kernel must have a bound, and the GPU checks its groups against [`most_groups`] before the
//! kernel runs, running none past it (gpu.rs, `Indirect`); the run then stops, naming the kernel.

use crate::error::{Error, Result};

/// The longest a single dispatch may hold the GPU, in milliseconds
/// ([`wrela_abi::STEP_MS`]).
pub const STEP_MS: f64 = wrela_abi::STEP_MS as f64;

/// The environment variable that lowers [`STEP_MS`] (never raises it).
pub const STEP_VAR: &str = "WRELA_GPU_STEP";

/// The most passes the compiler's count follows a loop for (`wrela_ir::work::MOST_PASSES`): a
/// loop that may go on past them has no bound.
const MOST_PASSES: u32 = 4096;

/// The most time one unit of a kernel's work bound takes on this machine's GPU, in nanoseconds
/// of a whole dispatch ([`wrela_abi::TargetLimits::step_femtoseconds`]): its time over its bound
/// times its invocations, measured on kernels that do all their bound's work. Real dispatches
/// do less than their bound, so take less; but reads that miss every cache take up to 100 times
/// as long a unit, so a kernel made of them can run past what this says (`Behind` is then the
/// second line).
pub fn unit_ns(target: wrela_abi::Target) -> f64 {
    f64::from(target.limits().step_femtoseconds) * 1e-6
}

/// How many invocations this machine's GPU runs side by side, about
/// ([`wrela_abi::TargetLimits::step_width`]): a dispatch of fewer takes as long as one of this
/// many, since one invocation alone runs its bound's work in a row.
pub fn width(target: wrela_abi::Target) -> u64 {
    u64::from(target.limits().step_width)
}

/// The limit in force: [`STEP_MS`], or less where [`STEP_VAR`] says so.
pub fn limit_ms() -> f64 {
    let lowered = std::env::var(STEP_VAR).ok().and_then(|v| v.trim().parse::<f64>().ok());
    match lowered {
        Some(ms) if ms > 0.0 && ms < STEP_MS => ms,
        _ => STEP_MS,
    }
}

/// The most work one dispatch may do here, in thousands of units, as a program reads it
/// (`std::gpu::limits().step_work`): the limit in force over [`unit_ns`].
pub fn step_work(target: wrela_abi::Target) -> u32 {
    let units = limit_ms() * 1e6 / unit_ns(target) / 1000.0;
    units.min(f64::from(u32::MAX)) as u32
}

/// Whether each dispatch and draw is written to stderr as it's recorded (`wrela-host --steps`):
/// its frame, label, pipeline, invocations (a draw's vertices and instances) and bound.
static LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Writes each dispatch and draw to stderr from now on (`wrela-host --steps`).
pub fn log_steps() {
    LOG.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`log_steps`] was asked for.
pub(crate) fn logging() -> bool {
    LOG.load(std::sync::atomic::Ordering::Relaxed)
}

/// What the build states of a compute kernel's work an invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Bound {
    /// At most this much work.
    Most(u64),
    /// No bound: the compiler's count found a loop with no exit it is sure of, there (or `""`
    /// where it didn't say).
    Unbounded(String),
    /// The build doesn't state it (a hand-written manifest without `work`).
    Unstated,
}

impl Bound {
    /// The bound a manifest states for a compute pipeline (its one entry point's).
    pub fn of(p: &wrela_abi::manifest::Pipeline) -> Bound {
        match p.work.first() {
            None => Bound::Unstated,
            Some(Some(w)) => Bound::Most(*w),
            Some(None) => {
                Bound::Unbounded(p.unbounded.first().cloned().flatten().unwrap_or_default())
            }
        }
    }

    /// As `--steps` writes it: its work, or `none`, or `unstated`.
    pub(crate) fn shown(&self) -> String {
        match self {
            Bound::Most(w) => w.to_string(),
            Bound::Unbounded(_) => "none".into(),
            Bound::Unstated => "unstated".into(),
        }
    }
}

/// The time `invocations` invocations of a kernel bounded by `most` each can take on this
/// machine, in milliseconds.
pub fn ms_of(most: u64, invocations: u64, target: wrela_abi::Target) -> f64 {
    most as f64 * invocations.max(width(target)) as f64 * unit_ns(target) * 1e-6
}

/// The most groups of `per_group` invocations a dispatch of a kernel bounded by `most` may run
/// within the limit: none where even one is past it.
pub fn most_groups(most: u64, per_group: u64, target: wrela_abi::Target) -> u64 {
    if ms_of(most, per_group, target) > limit_ms() {
        return 0;
    }
    let unit_ms = most.max(1) as f64 * per_group.max(1) as f64 * unit_ns(target) * 1e-6;
    (limit_ms() / unit_ms).floor().min(u64::MAX as f64) as u64
}

/// Whether a dispatch of kernel `name`, bounded by `bound`, of `invocations` invocations
/// (`None`: an indirect dispatch's, read on the GPU), may run: an error saying why not.
pub fn check(
    name: &str,
    bound: &Bound,
    invocations: Option<u64>,
    target: wrela_abi::Target,
) -> Result<()> {
    let limit = limit_ms();
    match bound {
        Bound::Unstated => Err(Error::Gpu(format!(
            "refused a dispatch of `{name}` before it ran: its build doesn't state the most work \
             an invocation of it can do (the manifest's `work`), so it could hold the GPU for \
             any time: build it with this repository's toolchain"
        ))),
        Bound::Unbounded(at) => Err(Error::Gpu(format!(
            "refused a dispatch of `{name}` before it ran: the compiler can't bound the work of \
             one invocation of it{}: a loop that may run past {MOST_PASSES} passes holds the GPU \
             for as long as it runs, and no dispatch may hold it past {limit} ms. Give the loop \
             a greatest length the compiler can see: clamp its count (`for i in 0..min(n, 64)`) \
             or what its condition compares with (docs/rates.md)",
            if at.is_empty() {
                String::new()
            } else {
                format!(" (the loop at {at} has no exit it is sure of)")
            },
        ))),
        Bound::Most(most) => {
            let Some(n) = invocations else { return Ok(()) };
            let ms = ms_of(*most, n, target);
            if ms <= limit {
                return Ok(());
            }
            let cut = match most_groups(*most, 1, target) {
                0 => "one invocation alone could take longer: bound its work lower".to_string(),
                k => format!(
                    "cut it into parts of at most {k} invocations (bands of rows, a slice of the \
                     cells), or bound an invocation's work lower"
                ),
            };
            Err(Error::Gpu(format!(
                "refused a dispatch of `{name}` before it ran: {n} invocations of at most {most} \
                 work each, up to {} ms on this GPU ({} ns a unit of work at most), over the \
                 {limit} ms a dispatch may hold it: {cut} (docs/rates.md)",
                shown_ms(ms),
                unit_ns(target),
            )))
        }
    }
}

/// Milliseconds as an error shows them: to a tenth under ten.
fn shown_ms(ms: f64) -> String {
    if ms < 10.0 { format!("{ms:.1}") } else { format!("{ms:.0}") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wrela_abi::Target;

    #[test]
    fn a_dispatch_is_refused_past_the_limit_and_when_its_kernel_is_unbounded() {
        for t in [Target::Metal, Target::Vulkan] {
            let most = most_groups(1000, 1, t);
            assert!(most > width(t), "{t:?}: {most}");
            assert!(check("k", &Bound::Most(1000), Some(most), t).is_ok());
            let e = check("k", &Bound::Most(1000), Some(most + 1), t).unwrap_err().to_string();
            assert!(e.contains("refused a dispatch of `k`") && e.contains("1000 work each"), "{e}");
            let e = check("k", &Bound::Unbounded("a.wrela:3:5".into()), Some(1), t);
            assert!(e.unwrap_err().to_string().contains("the loop at a.wrela:3:5"));
            assert!(check("k", &Bound::Unstated, Some(1), t).is_err());
            // An indirect dispatch's invocations aren't known here: its kernel needs a bound.
            assert!(check("k", &Bound::Most(u64::MAX), None, t).is_ok());
            // One invocation alone runs its whole bound in a row: as long as a GPU's width.
            let long = (STEP_MS * 1e6 / (unit_ns(t) * width(t) as f64)) as u64 * 2;
            assert!(check("k", &Bound::Most(long), Some(1), t).is_err());
            assert_eq!(most_groups(long, 1, t), 0);
        }
    }
}

/// The GPU's check of indirect dispatches' groups ([`crate::step`]): each indirect dispatch's
/// arguments are first read by a kernel of the host's own, which writes them where the dispatch
/// then reads them, or none (no groups) where they're past the most the kernel's bound allows,
/// and marks the dispatch cut. A frame's marks are read back once its work is done: a cut
/// dispatch stops the run with an error naming its kernel. A dispatch the check passes runs as
/// the program asked; one it cuts never runs.
pub(crate) struct Indirect {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// Checked arguments, 16 bytes a dispatch this frame (x, y, z, cut).
    args: wgpu::Buffer,
    /// Each check's numbers, [`CHECK_STRIDE`] bytes a dispatch this frame.
    checks: wgpu::Buffer,
    /// How many dispatches `args` and `checks` hold.
    capacity: u32,
    /// The kernels dispatched indirectly this frame, in their slots' order, with what the
    /// program's buffer was bound as for the check.
    names: Vec<String>,
    /// Bind groups by the program's buffer they read (its handle), for the frame's `args`.
    groups: std::collections::HashMap<u32, wgpu::BindGroup>,
    /// Frames' marks being read back: the buffer, its kernels, and whether it has been mapped
    /// (0 not yet asked, 3 asked, 1 mapped, 2 failed).
    pending: std::collections::VecDeque<(
        wgpu::Buffer,
        Vec<String>,
        std::sync::Arc<std::sync::atomic::AtomicU8>,
    )>,
}

/// An indirect dispatch's check, got ready ([`Indirect::prepare`]).
pub(crate) struct Checked {
    group: wgpu::BindGroup,
    offset: u32,
    pub(crate) args: wgpu::Buffer,
    pub(crate) at: u64,
}

/// The most indirect dispatches a frame may make.
const MOST_INDIRECT: u32 = 4096;

/// The bytes between two dispatches' check numbers (a uniform's dynamic offset's alignment).
const CHECK_STRIDE: u64 = 256;

const CHECK_WGSL: &str = "
struct Check { at: u32, slot: u32, most: u32, unused: u32 }
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> args: array<u32>;
@group(0) @binding(2) var<uniform> c: Check;
@compute @workgroup_size(1) fn checked() {
    let x = src[c.at];
    let y = src[c.at + 1u];
    let z = src[c.at + 2u];
    var over = false;
    if x != 0u && y != 0u && z != 0u {
        if x > c.most {
            over = true;
        } else if y > c.most / x {
            over = true;
        } else if z > c.most / (x * y) {
            over = true;
        }
    }
    let o = c.slot * 4u;
    if over {
        args[o] = 0u; args[o + 1u] = 0u; args[o + 2u] = 0u; args[o + 3u] = 1u;
    } else {
        args[o] = x; args[o + 1u] = y; args[o + 2u] = z; args[o + 3u] = 0u;
    }
}
";

impl Indirect {
    pub(crate) fn new(device: &wgpu::Device) -> Indirect {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("indirect check"),
            source: wgpu::ShaderSource::Wgsl(CHECK_WGSL.into()),
        });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("indirect check"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(false)),
                entry(
                    2,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("indirect check"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("indirect check"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("checked"),
            compilation_options: Default::default(),
            cache: None,
        });
        let (args, checks) = Indirect::buffers(device, MOST_INDIRECT);
        Indirect {
            pipeline,
            layout,
            args,
            checks,
            capacity: MOST_INDIRECT,
            names: Vec::new(),
            groups: std::collections::HashMap::new(),
            pending: std::collections::VecDeque::new(),
        }
    }

    fn buffers(device: &wgpu::Device, n: u32) -> (wgpu::Buffer, wgpu::Buffer) {
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect checked"),
            size: u64::from(n) * 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let checks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect checks"),
            size: u64::from(n) * CHECK_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (args, checks)
    }

    /// Gets ready the check of an indirect dispatch of kernel `name` whose arguments are at byte
    /// `at` of `source` (the program's buffer `handle`), at most `most` groups: what
    /// [`Indirect::record`] records, in the dispatch's own pass, before the dispatch.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        name: &str,
        (handle, source, at): (u32, &wgpu::Buffer, u32),
        most: u64,
    ) -> Result<Checked> {
        let slot = self.names.len() as u32;
        if slot >= self.capacity {
            return Err(Error::Gpu(format!(
                "refused an indirect dispatch of `{name}`: a frame may make at most {} \
                 indirect dispatches, each checked on the GPU before it runs",
                self.capacity
            )));
        }
        let numbers = [at / 4, slot, most.min(u64::from(u32::MAX)) as u32, 0];
        let bytes: Vec<u8> = numbers.iter().flat_map(|n| n.to_le_bytes()).collect();
        queue.write_buffer(&self.checks, u64::from(slot) * CHECK_STRIDE, &bytes);
        let layout = &self.layout;
        let (args, checks) = (&self.args, &self.checks);
        let group = self.groups.entry(handle).or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("indirect check"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: source.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: args.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: checks,
                            offset: 0,
                            size: wgpu::BufferSize::new(16),
                        }),
                    },
                ],
            })
        });
        self.names.push(name.to_string());
        Ok(Checked {
            group: group.clone(),
            offset: (u64::from(slot) * CHECK_STRIDE) as u32,
            args: self.args.clone(),
            at: u64::from(slot) * 16,
        })
    }

    /// Records the check `c` in `pass`, before its dispatch: the dispatch then reads its groups
    /// from `c.args` at `c.at`.
    pub(crate) fn record(&self, pass: &mut wgpu::ComputePass<'_>, c: &Checked) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &c.group, &[c.offset]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Ends a frame: its marks are copied out to be read once its work is done.
    pub(crate) fn end_frame(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if self.names.is_empty() {
            return;
        }
        let names = std::mem::take(&mut self.names);
        let size = names.len() as u64 * 16;
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect marks"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.args, 0, &read, 0, size);
        self.pending.push_back((
            read,
            names,
            std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)),
        ));
        self.groups.clear();
    }

    /// After a frame was submitted: asks for its marks, and fails if a frame already read cut
    /// a dispatch.
    pub(crate) fn read(&mut self) -> Result<()> {
        use std::sync::atomic::Ordering;
        for (buffer, _, state) in &self.pending {
            if state.load(Ordering::Acquire) == 0 {
                state.store(3, Ordering::Release);
                let s = state.clone();
                buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    s.store(if r.is_ok() { 1 } else { 2 }, Ordering::Release);
                });
            }
        }
        while let Some((buffer, names, state)) = self.pending.front() {
            match state.load(Ordering::Acquire) {
                0 | 3 => break,
                2 => {
                    return Err(Error::Gpu(
                        "reading back the indirect dispatches' checks failed".into(),
                    ));
                }
                _ => {}
            }
            let cut: Vec<&str> = {
                let mapped =
                    buffer.slice(..).get_mapped_range().map_err(|e| Error::Gpu(e.to_string()))?;
                mapped
                    .chunks_exact(16)
                    .zip(names)
                    .filter(|(a, _)| a[12..16] != [0, 0, 0, 0])
                    .map(|(_, n)| n.as_str())
                    .collect()
            };
            if let Some(name) = cut.first() {
                return Err(Error::Gpu(format!(
                    "refused an indirect dispatch of `{name}` before it ran: the groups the GPU \
                     worked out for it were past the most its kernel's bound allows within the \
                     {} ms a dispatch may hold the GPU; it ran none. Cut it into parts, or bound \
                     an invocation's work lower (docs/rates.md)",
                    limit_ms()
                )));
            }
            buffer.unmap();
            self.pending.pop_front();
        }
        Ok(())
    }
}
