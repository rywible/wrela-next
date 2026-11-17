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
    pipeline: wrela_gpu::ComputePipeline,
    layout: wrela_gpu::BindGroupLayout,
    /// The check's manifest entry: where its shader reads its buffers' sizes.
    entry: wrela_abi::manifest::Pipeline,
    /// Checked arguments, 16 bytes a dispatch this frame (x, y, z, cut).
    args: wrela_gpu::Buffer,
    /// Each check's numbers, [`CHECK_STRIDE`] bytes a dispatch this frame.
    checks: wrela_gpu::Buffer,
    /// How many dispatches `args` and `checks` hold.
    capacity: u32,
    /// The kernels dispatched indirectly this frame, in their slots' order.
    names: Vec<String>,
    /// Bind groups by the program's buffer they read (its handle), for the frame's `args`.
    groups: std::collections::HashMap<u32, wrela_gpu::BindGroup>,
    /// Frames' marks being read back: the buffer, its kernels, and the submission that wrote
    /// them (once it's made).
    pending: std::collections::VecDeque<(
        wrela_gpu::Buffer,
        Vec<String>,
        Option<wrela_gpu::SubmissionIndex>,
    )>,
}

/// An indirect dispatch's check, got ready ([`Indirect::prepare`]).
pub(crate) struct Checked {
    group: wrela_gpu::BindGroup,
    offset: u32,
    pub(crate) args: wrela_gpu::Buffer,
    pub(crate) at: u64,
}

/// The most indirect dispatches a frame may make.
const MOST_INDIRECT: u32 = 4096;

/// The bytes between two dispatches' check numbers (a uniform's dynamic offset's alignment).
const CHECK_STRIDE: u64 = 256;

/// The check's kernel, as IR, written by the compiler's own writers for this machine:
/// ```text
/// struct Check { at: u32, slot: u32, most: u32, unused: u32 }
/// @group(0) @binding(0) var<storage, read> src: array<u32>;
/// @group(0) @binding(1) var<storage, read_write> args: array<u32>;
/// @group(0) @binding(2) var<uniform> c: Check;
/// @compute @workgroup_size(1) fn checked() {
///     let (x, y, z) = (src[c.at], src[c.at + 1], src[c.at + 2]);
///     let over = x != 0 && y != 0 && z != 0
///         && (x > c.most || y > c.most / x || z > c.most / (x * y));
///     let o = c.slot * 4;
///     args[o..o + 4] = if over { (0, 0, 0, 1) } else { (x, y, z, 0) };
/// }
/// ```
/// (Where `x > c.most` the other two may divide by zero or wrap: the writers' guarded division
/// gives a value, and `over` is already true.)
fn check_kernel() -> wrela_shader::build::Builder {
    use wrela_shader::build::{BinOp, Builder, Proj, Scalar};
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let check = b.structure("Check", &[("at", u), ("slot", u), ("most", u), ("unused", u)]);
    let src = b.storage("src", 0, u, false);
    let args = b.storage("args", 1, u, true);
    let c = b.uniform("c", 2, check);
    b.compute("checked", [1, 1, 1], &[], |f| {
        let field = |f: &mut wrela_shader::build::Body<'_>, k: u32| {
            f.load(u, Builder::whole(c).with(Proj::Field(k)))
        };
        let (at, slot, most) = (field(f, 0), field(f, 1), field(f, 2));
        let read = |f: &mut wrela_shader::build::Body<'_>, k: u32| {
            let k = f.u32(k);
            let i = f.binary(BinOp::Add, at, k);
            f.load(u, Builder::element(src, i))
        };
        let (x, y, z) = (read(f, 0), read(f, 1), read(f, 2));
        let zero = f.u32(0);
        let nx = f.binary(BinOp::Ne, x, zero);
        let ny = f.binary(BinOp::Ne, y, zero);
        let nz = f.binary(BinOp::Ne, z, zero);
        let nxy = f.binary(BinOp::And, nx, ny);
        let nonzero = f.binary(BinOp::And, nxy, nz);
        let over_x = f.binary(BinOp::Gt, x, most);
        let per_x = f.binary(BinOp::Div, most, x);
        let over_y = f.binary(BinOp::Gt, y, per_x);
        let xy = f.binary(BinOp::Mul, x, y);
        let per_xy = f.binary(BinOp::Div, most, xy);
        let over_z = f.binary(BinOp::Gt, z, per_xy);
        let over_xy = f.binary(BinOp::Or, over_x, over_y);
        let any = f.binary(BinOp::Or, over_xy, over_z);
        let over = f.binary(BinOp::And, nonzero, any);
        let four = f.u32(4);
        let o = f.binary(BinOp::Mul, slot, four);
        let one = f.u32(1);
        for (k, kept, cut) in [(0, x, zero), (1, y, zero), (2, z, zero), (3, zero, one)] {
            let v = f.select(over, cut, kept);
            let kk = f.u32(k);
            let i = f.binary(BinOp::Add, o, kk);
            f.store(Builder::element(args, i), v);
        }
    });
    b
}

/// The check's manifest entry: its uniform block and its two buffers.
fn check_entry() -> wrela_abi::manifest::Pipeline {
    use wrela_abi::manifest::{
        BindingKind, BindingStage, Pipeline, ResourceBinding, Stage, UniformBlock, UniformSpace,
    };
    let binding =
        |binding, kind| ResourceBinding { binding, kind, stage: BindingStage::Both, format: None };
    Pipeline {
        name: "indirect check".into(),
        shader: "indirect check".into(),
        stage: Stage::Compute { entry: "checked".into(), workgroup_size: [1, 1, 1] },
        uniform: Some(UniformBlock { binding: 2, size: 16, space: UniformSpace::Uniform }),
        bindings: vec![binding(0, BindingKind::Read), binding(1, BindingKind::ReadWrite)],
        debug_flag: None,
        requires: Vec::new(),
        conditional: false,
        standin: None,
        sizes: None,
        kept: 0,
        work: vec![Some(64)],
        unbounded: Vec::new(),
    }
}

impl Indirect {
    pub(crate) fn new(device: &wrela_gpu::Device) -> Result<Indirect> {
        let mut entry = check_entry();
        let shader = check_kernel()
            .write(&mut entry, wrela_abi::Target::HERE)
            .map_err(|e| Error::Gpu(format!("the indirect check's shader: {e}")))?;
        let entries: Vec<wrela_gpu::LayoutEntry> = wrela_abi::metal::layout(&entry)
            .iter()
            .map(|e| wrela_gpu::LayoutEntry {
                binding: e.binding,
                visibility: e.visibility,
                ty: crate::gpu::binding_type(e, &entry),
            })
            .collect();
        let layout = device.create_bind_group_layout(&entries);
        let module = device
            .create_shader_module("indirect check", &shader.bytes)
            .map_err(|e| Error::Gpu(format!("the indirect check's shader: {}", e.0)))?;
        let wrela_abi::manifest::Stage::Compute { entry: name, .. } = &entry.stage else {
            unreachable!("a kernel");
        };
        let pipeline = device
            .create_compute_pipeline(&wrela_gpu::ComputePipelineDesc {
                label: "indirect check",
                layout: &layout,
                module: &module,
                entry: name,
                workgroup_size: [1, 1, 1],
            })
            .map_err(|e| Error::Gpu(format!("the indirect check's pipeline: {}", e.0)))?;
        let (args, checks) = Indirect::buffers(device, MOST_INDIRECT);
        Ok(Indirect {
            pipeline,
            layout,
            entry,
            args,
            checks,
            capacity: MOST_INDIRECT,
            names: Vec::new(),
            groups: std::collections::HashMap::new(),
            pending: std::collections::VecDeque::new(),
        })
    }

    fn buffers(device: &wrela_gpu::Device, n: u32) -> (wrela_gpu::Buffer, wrela_gpu::Buffer) {
        let args = device.create_buffer(&wrela_gpu::BufferDesc {
            label: "indirect checked",
            size: u64::from(n) * 16,
            mappable: false,
        });
        let checks = device.create_buffer(&wrela_gpu::BufferDesc {
            label: "indirect checks",
            size: u64::from(n) * CHECK_STRIDE,
            mappable: false,
        });
        (args, checks)
    }

    /// Gets ready the check of an indirect dispatch of kernel `name` whose arguments are at byte
    /// `at` of `source` (the program's buffer `handle`), at most `most` groups: what
    /// [`Indirect::record`] records, in a pass of its own just before the dispatch's.
    pub(crate) fn prepare(
        &mut self,
        device: &wrela_gpu::Device,
        name: &str,
        (handle, source, at): (u32, &wrela_gpu::Buffer, u32),
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
        device.write_buffer(&self.checks, u64::from(slot) * CHECK_STRIDE, &bytes);
        let (layout, entry) = (&self.layout, &self.entry);
        let (args, checks) = (&self.args, &self.checks);
        let group = self.groups.entry(handle).or_insert_with(|| {
            let mut entries = vec![
                wrela_gpu::BindGroupEntry {
                    binding: 0,
                    resource: wrela_gpu::Resource::Buffer { buffer: source, offset: 0, size: None },
                },
                wrela_gpu::BindGroupEntry {
                    binding: 1,
                    resource: wrela_gpu::Resource::Buffer { buffer: args, offset: 0, size: None },
                },
                wrela_gpu::BindGroupEntry {
                    binding: 2,
                    resource: wrela_gpu::Resource::Buffer {
                        buffer: checks,
                        offset: 0,
                        size: Some(16),
                    },
                },
            ];
            // Metal's shader reads its buffers' sizes from a uniform of their own.
            let sizes = entry.sizes.as_ref().map(|s| {
                let words: Vec<u32> = s
                    .buffers
                    .iter()
                    .map(|&b| match b {
                        0 => source.size() as u32,
                        1 => args.size() as u32,
                        _ => 0,
                    })
                    .collect();
                let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                bytes.resize(bytes.len().next_multiple_of(16).max(16), 0);
                let b = device.create_buffer(&wrela_gpu::BufferDesc {
                    label: "indirect check sizes",
                    size: bytes.len() as u64,
                    mappable: false,
                });
                device.write_buffer(&b, 0, &bytes);
                (s.binding, b)
            });
            if let Some((binding, b)) = &sizes {
                entries.push(wrela_gpu::BindGroupEntry {
                    binding: *binding,
                    resource: wrela_gpu::Resource::Buffer { buffer: b, offset: 0, size: None },
                });
            }
            device.create_bind_group(layout, &entries)
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
    pub(crate) fn record(&self, pass: &mut wrela_gpu::ComputePass<'_>, c: &Checked) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(&c.group, &[c.offset]);
        pass.dispatch(1, 1, 1);
    }

    /// Ends a frame: its marks are copied out to be read once its work is done.
    pub(crate) fn end_frame(
        &mut self,
        device: &wrela_gpu::Device,
        encoder: &mut wrela_gpu::CommandEncoder,
    ) {
        if self.names.is_empty() {
            return;
        }
        let names = std::mem::take(&mut self.names);
        let size = names.len() as u64 * 16;
        let read = device.create_buffer(&wrela_gpu::BufferDesc {
            label: "indirect marks",
            size,
            mappable: true,
        });
        encoder.copy_buffer_to_buffer(&self.args, 0, &read, 0, size);
        self.pending.push_back((read, names, None));
        self.groups.clear();
    }

    /// After a frame was submitted: notes its submission, and fails if a frame whose work is
    /// done cut a dispatch.
    pub(crate) fn read(&mut self, device: &wrela_gpu::Device) -> Result<()> {
        for (_, _, at) in &mut self.pending {
            at.get_or_insert(device.submitted());
        }
        let done = device.completed();
        while let Some((buffer, names, Some(at))) = self.pending.front() {
            if *at > done {
                break;
            }
            let bytes = buffer.read(0, buffer.size()).map_err(|e| Error::Gpu(e.0))?;
            let cut: Vec<&str> = bytes
                .chunks_exact(16)
                .zip(names)
                .filter(|(a, _)| a[12..16] != [0, 0, 0, 0])
                .map(|(_, n)| n.as_str())
                .collect();
            if let Some(name) = cut.first() {
                return Err(Error::Gpu(format!(
                    "refused an indirect dispatch of `{name}` before it ran: the groups the GPU \
                     worked out for it were past the most its kernel's bound allows within the \
                     {} ms a dispatch may hold the GPU; it ran none. Cut it into parts, or bound \
                     an invocation's work lower (docs/rates.md)",
                    limit_ms()
                )));
            }
            self.pending.pop_front();
        }
        Ok(())
    }
}
