//! The GPU side: carries out decoded commands on the machine's own GPU API (`wrela_gpu`: Metal
//! or Vulkan), headless.
//!
//! The "screen" is an offscreen [`wrela_abi::manifest::SCREEN_FORMAT`] texture that can be read
//! back. How commands map to the GPU (the browser runtime's decoder does the same with WebGPU):
//!
//! - Work is recorded into one pending command encoder and submitted ("flushed") at the end of
//!   a pass on the screen, before a large upload (so earlier work sees the old contents), before
//!   a readback, and at the end of each call. Smaller buffer and texture writes use ordered
//!   copies from an upload ring. Each dispatch gets its own compute pass.
//! - A pass's draws are collected and recorded into one render pass when it ends (`Present` or
//!   `EndPass`), since a pass can span batches. A pass that may join the one before (`join`), and
//!   begins right after it ended on the same targets, keeping what's in them, under the same
//!   label (a `Label` between ends the first), runs as part of it, unless the timing is serial:
//!   on a tile-based GPU each pass stores its targets to memory and the next loads them back.
//!   Only a pass the program marks: a pass's geometry is binned before its tiles are drawn, so
//!   two heavy passes made one wait longer than they save.
//! - A render pipeline is made at load for each of its manifest's `targets` (the passes the
//!   program draws it in): the screen's or a texture's colour format, and a depth format or
//!   none. With depth, a fragment is kept where its depth is less than what's there, which it
//!   replaces.
//! - Uniform bytes go into a ring buffer bound with dynamic offsets (aligned to the device's
//!   offset alignment, 256 by default). The ring is staged on the CPU and written just before
//!   each flush, so every command in a submission has its own slice. It grows (after a flush)
//!   when one submission needs more than it holds.
//! - Bind group 0 of each pipeline is made from the manifest: the uniform block (dynamic offset)
//!   and the bindings, in order. Bind groups are cached per (pipeline, bindings), until one of
//!   their resources is destroyed.
//! - Destroying a resource drops this host's reference to it; `wrela_gpu` keeps it alive for
//!   work already recorded.
//!
//! Commands arrive decoded, sequenced and checked ([`wrela_abi::check`]); an error the GPU
//! reports here (Vulkan's validation layer, a failed command buffer) is a host bug, reported as
//! [`Error::Gpu`].

use crate::Timing;
use crate::error::{Error, Result};
use crate::program::Executor;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use wrela_abi::hash::StateHash;
use wrela_abi::manifest::{
    BindingKind, Blend, Cull, DepthBias, DepthState, Manifest, ResourceBinding, Stage, UniformSpace,
};
use wrela_abi::stream::{
    self, Binding, Command, Compare, Opcode, Pass, TexelRect, TextureFormat, TextureKind, View,
};

mod calibrate;
mod present;
#[cfg(test)]
mod strict_ops;
#[cfg(test)]
mod tests;

pub use calibrate::keep_clocked;
#[cfg(test)]
use calibrate::measured_unit;
use calibrate::{Warmer, ensure_unit};
use present::{Presenter, draw_presented, present_pipeline};

/// The screen's format: `SCREEN_FORMAT` (rgba8unorm).
const SCREEN_FORMAT: TextureFormat = wrela_gpu::SCREEN_FORMAT;

/// `x` rounded up to a multiple of `alignment`.
fn align_to(x: u64, alignment: u64) -> u64 {
    x.div_ceil(alignment) * alignment
}
/// The uniform ring's starting size.
const RING_START: u64 = 64 * 1024;
/// Timestamp queries the timer starts with (two per timed pass).
const TIMESTAMP_QUERIES: u32 = 512;
/// The most the timer grows to: 2,048 timed passes between two readings (32 KB of samples, the
/// least a Metal counter sample buffer is held to).
const TIMESTAMP_QUERIES_MOST: u32 = 4096;

/// One timed piece of GPU work, from timestamp queries.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuTiming {
    /// Index of the frame in the run.
    pub frame: usize,
    /// The name the program gave it (`std::gpu::label`), else the dispatch's pipeline name, or
    /// `screen pass` or `pass`.
    pub label: String,
    /// Its duration, `end - start`.
    pub nanos: f64,
    /// When it started and ended on the GPU's clock, in nanoseconds from an origin of its own:
    /// a frame's span is its first start to its last end.
    pub start: f64,
    pub end: f64,
    /// Which submission it ran in, counted from 0 over the run (the serial mode submits each
    /// pass alone).
    pub submission: usize,
}

/// Each frame's span in `timings` (ms): from its first timed piece's start to its last's end,
/// by frame, for the frames that have any.
pub fn frame_spans(timings: &[GpuTiming]) -> Vec<(usize, f64)> {
    let mut spans: std::collections::BTreeMap<usize, (f64, f64)> = Default::default();
    for t in timings {
        let e = spans.entry(t.frame).or_insert((t.start, t.end));
        e.0 = e.0.min(t.start);
        e.1 = e.1.max(t.end);
    }
    spans.into_iter().map(|(k, (a, b))| (k, (b - a) / 1e6)).collect()
}

/// A gap in a frame's timed work: its length (ms) and the labels of the pieces that end before
/// it and begin after it.
pub type Gap = (f64, String, String);

/// Each frame's idle time inside its span (ms: the span less the union of its timed pieces), and
/// its longest gaps, longest first ([`GAPS_SAID`] at most). What a span holds besides work: the
/// GPU waiting on the CPU between submissions.
pub fn frame_gaps(timings: &[GpuTiming]) -> Vec<(usize, f64, Vec<Gap>)> {
    let mut by: std::collections::BTreeMap<usize, Vec<&GpuTiming>> = Default::default();
    for t in timings {
        by.entry(t.frame).or_default().push(t);
    }
    let mut out = Vec::new();
    for (frame, mut ts) in by {
        ts.sort_by(|a, b| a.start.total_cmp(&b.start));
        let mut end = ts[0].end;
        let mut end_label = ts[0].label.clone();
        let mut idle = 0.0;
        let mut gaps: Vec<Gap> = Vec::new();
        for t in &ts[1..] {
            if t.start > end {
                let g = t.start - end;
                idle += g;
                gaps.push((g / 1e6, end_label.clone(), t.label.clone()));
            }
            if t.end > end {
                end = t.end;
                end_label = t.label.clone();
            }
        }
        gaps.sort_by(|a, b| b.0.total_cmp(&a.0));
        gaps.truncate(GAPS_SAID);
        out.push((frame, idle / 1e6, gaps));
    }
    out
}

/// How many of a frame's longest gaps `frame_gaps` says.
pub const GAPS_SAID: usize = 5;

/// A copy's extent in whole blocks of `format`: a rectangle of a compressed texture that reaches
/// its level's edge (the checker allows no other part block) covers that edge's last block, which
/// WebGPU counts in the level's physical size (a 4 × 4 BC level 1 is one 4 × 4 block).
fn whole_blocks(format: TextureFormat, width: u32, height: u32) -> [u32; 3] {
    let b = format.block();
    [width.div_ceil(b) * b, height.div_ceil(b) * b, 1]
}

/// One texel of `format` (its bytes, little-endian) as RGBA8, as [`Gpu::read_target`] gives it.
fn as_rgba8(format: TextureFormat, t: &[u8]) -> [u8; 4] {
    let unit = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
    let half = |i: usize| f16_to_f32(u16::from_le_bytes([t[i], t[i + 1]]));
    let single = |i: usize| f32::from_le_bytes([t[i], t[i + 1], t[i + 2], t[i + 3]]);
    match format {
        TextureFormat::Rgba8 => [t[0], t[1], t[2], t[3]],
        TextureFormat::Rgba16Float => [unit(half(0)), unit(half(2)), unit(half(4)), unit(half(6))],
        TextureFormat::Rg16Float => [unit(half(0)), unit(half(2)), 0, 255],
        TextureFormat::R16Float => {
            let g = unit(half(0));
            [g, g, g, 255]
        }
        TextureFormat::R32Float | TextureFormat::Depth32Float => {
            let g = unit(single(0));
            [g, g, g, 255]
        }
        TextureFormat::R32Uint | TextureFormat::Rg32Uint => [0, 0, 0, 255],
        TextureFormat::R8 => [t[0], t[0], t[0], 255],
        TextureFormat::Rg8 => [t[0], t[1], 0, 255],
        TextureFormat::Rgb10a2 => {
            let w = u32::from_le_bytes([t[0], t[1], t[2], t[3]]);
            let c = |shift: u32| unit((w >> shift & 1023) as f32 / 1023.0);
            [c(0), c(10), c(20), unit((w >> 30) as f32 / 3.0)]
        }
        TextureFormat::Rg32Float => [unit(single(0)), unit(single(4)), 0, 255],
        TextureFormat::Rgba32Float => {
            [unit(single(0)), unit(single(4)), unit(single(8)), unit(single(12))]
        }
        TextureFormat::Rg11b10Float => {
            let w = u32::from_le_bytes([t[0], t[1], t[2], t[3]]);
            let small = |bits: u32, mantissa: u32| {
                let e = (bits >> mantissa) as i32;
                let m = (bits & ((1 << mantissa) - 1)) as f32 / (1 << mantissa) as f32;
                match e {
                    0 => m * 2f32.powi(-14),
                    31 => f32::INFINITY,
                    e => (1.0 + m) * 2f32.powi(e - 15),
                }
            };
            [unit(small(w & 2047, 6)), unit(small(w >> 11 & 2047, 6)), unit(small(w >> 22, 5)), 255]
        }
        // A compressed format is never a pass's target, which is what's read back.
        TextureFormat::Bc1
        | TextureFormat::Bc4
        | TextureFormat::Bc5
        | TextureFormat::Bc7
        | TextureFormat::Etc2Rgb8
        | TextureFormat::Astc4x4 => [0, 0, 0, 255],
    }
}

/// An IEEE half's value.
fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((h >> 10) & 0x1f);
    let fraction = f32::from(h & 0x3ff);
    match exponent {
        0 => sign * fraction * 2f32.powi(-24),
        31 if fraction == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        e => sign * (1.0 + fraction / 1024.0) * 2f32.powi(e - 15),
    }
}

/// FNV-1a 64 of a pipeline's manifest entry and its WGSL: what a hot reload compares, rather
/// than keeping a copy of every shader. Not its WGSL file's name, which is its place among the
/// build's pipelines (`pipeline_3.metal`): a pipeline added or dropped before it moves it, and the
/// next build keeps it all the same.
pub(crate) fn pipeline_key(p: &wrela_abi::manifest::Pipeline, source: &[u8]) -> u64 {
    let p = wrela_abi::manifest::Pipeline { shader: String::new(), ..p.clone() };
    let mut hash = StateHash::new();
    hash.update(format!("{p:?}\n").as_bytes());
    hash.update(source);
    hash.value()
}

/// A colour texture binding's format, which the manifest's parser requires of it.
fn colour_format(b: &ResourceBinding) -> TextureFormat {
    b.format.expect("the manifest's parser gives every colour texture binding its format")
}

/// A texture binding's view dimension.
fn view_dimension(kind: BindingKind) -> wrela_gpu::ViewDimension {
    match kind.view_kind() {
        Some(TextureKind::D3) => wrela_gpu::ViewDimension::D3,
        Some(TextureKind::Array) => wrela_gpu::ViewDimension::D2Array,
        Some(TextureKind::Cube) => wrela_gpu::ViewDimension::Cube,
        Some(TextureKind::D2) | None => wrela_gpu::ViewDimension::D2,
    }
}

/// The feature bits a program sees (`wrela_abi::Limits::features`) of a device with these
/// features: the required ones, and the rest only where `kept` (test mode's mask: a program's
/// fallback paths, measured).
fn feature_bits(features: &wrela_gpu::Features, kept: impl Fn(&str) -> bool) -> u32 {
    wrela_abi::FEATURES
        .iter()
        .filter(|name| features.has(name))
        .filter_map(|name| {
            let bit = wrela_abi::feature_bit(name)?;
            (bit & wrela_abi::REQUIRED_FEATURES != 0 || kept(name)).then_some(bit)
        })
        .fold(0, |a, b| a | b)
}

/// How a shader reads a colour texture of format `f`: filtered floats, unfiltered floats, or
/// unsigned integers.
fn sample_type(f: TextureFormat) -> wrela_gpu::SampleType {
    if f.is_uint() {
        wrela_gpu::SampleType::Uint
    } else {
        wrela_gpu::SampleType::Float { filterable: f.filterable() }
    }
}

/// A live resource the program made.
enum Resource {
    Buffer(wrela_gpu::Buffer),
    Texture(Texture),
    Sampler(wrela_gpu::Sampler),
}

/// A live texture, and the views of it that bindings and passes have asked for.
struct Texture {
    texture: wrela_gpu::Texture,
    format: TextureFormat,
    /// Its views, by the view and its dimension: made once each.
    views:
        std::cell::RefCell<HashMap<(View, wrela_gpu::ViewDimension, bool), wrela_gpu::TextureView>>,
}

impl Texture {
    /// A view of `view`'s levels and layers in `dimension` (`storage`: one level, the first if
    /// it names none, as a storage binding takes).
    fn view(
        &self,
        view: View,
        dimension: wrela_gpu::ViewDimension,
        storage: bool,
    ) -> wrela_gpu::TextureView {
        self.views
            .borrow_mut()
            .entry((view, dimension, storage))
            .or_insert_with(|| {
                let level = view.level.or(storage.then_some(0));
                self.texture.create_view(&wrela_gpu::ViewDesc {
                    dimension,
                    base_level: level.unwrap_or(0),
                    level_count: level.map(|_| 1),
                    base_layer: view.layer.unwrap_or(0),
                    layer_count: view.layer.map(|_| 1),
                })
            })
            .clone()
    }

    /// The 2D view of one level and layer: a pass's target.
    fn target(&self, level: u32, layer: u32) -> wrela_gpu::TextureView {
        let dimension = wrela_gpu::ViewDimension::D2;
        self.view(View { level: Some(level), layer: Some(layer) }, dimension, false)
    }
}

/// The target formats a render pipeline is made for: its colour formats (none in a pass that
/// draws only depths), and a depth format (or none).
type Targets = (Vec<TextureFormat>, Option<TextureFormat>);

pub(crate) struct Pipeline {
    name: Arc<str>,
    kind: Kind,
    layout: wrela_gpu::BindGroupLayout,
    /// The uniform block's binding and size, if it has one.
    uniform: Option<(u32, u32)>,
    bindings: Vec<ResourceBinding>,
    /// A debug build's: where its bounds checks' flag is bound.
    debug_flag: Option<u32>,
    /// Where its shader reads its storage buffers' sizes, and whose.
    sizes: Option<wrela_abi::manifest::BufferSizes>,
    /// A compute pipeline's most work an invocation, as its build states it, which a dispatch is
    /// weighed by before it runs (`crate::step`).
    bound: crate::step::Bound,
    /// A compute pipeline's invocations a workgroup (a render pipeline's is 1).
    per_group: u64,
    /// A render pipeline's screen targets: whether each has depth. An HDR screen draws them
    /// with its own colour format.
    screen_depths: Vec<bool>,
}

enum Kind {
    Compute(wrela_gpu::ComputePipeline),
    Render {
        module: wrela_gpu::ShaderModule,
        vertex: String,
        fragment: String,
        /// How each colour output meets its target's (the manifest's `blends`).
        blends: Vec<Blend>,
        /// The manifest's `cull`, `depth_bias` and `depth`.
        cull: Cull,
        depth_bias: DepthBias,
        depth: DepthState,
        variants: HashMap<Targets, wrela_gpu::RenderPipeline>,
    },
    /// Its shader needs device features this device hasn't (the manifest's `requires`): the
    /// checker refuses a command that records it.
    Unavailable,
}

/// The names of the passes the program named none: on the screen, and into textures.
static UNNAMED_PASSES: std::sync::LazyLock<(Arc<str>, Arc<str>)> =
    std::sync::LazyLock::new(|| (Arc::from("screen pass"), Arc::from("pass")));

/// A cached bind group: its pipeline, and the bindings it holds.
type BindGroupKey = (u32, Arc<[Binding]>);

enum DrawHow {
    Direct {
        vertices: u32,
        instances: u32,
    },
    Indirect {
        buffer: u32,
        offset: u32,
    },
    /// `u32` indices from `at` in buffer `indices`.
    IndexedIndirect {
        indices: u32,
        at: u32,
        buffer: u32,
        offset: u32,
    },
    Indexed {
        indices: u32,
        at: u32,
        count: u32,
        instances: u32,
        first: u32,
        base_vertex: i32,
        first_instance: u32,
    },
    /// `count` indirect draws, their arguments one after another from `offset`, indexed when
    /// it has `indices` (`[handle, at]`).
    Multi {
        indices: Option<[u32; 2]>,
        buffer: u32,
        offset: u32,
        count: u32,
    },
}

/// What a pass records, in order: draws, and the scissors and viewports set between them.
enum PassItem {
    Draw(PendingDraw),
    Scissor([u32; 4]),
    Viewport([f32; 6]),
}

struct PendingDraw {
    pipeline: u32,
    how: DrawHow,
    bindings: Vec<Binding>,
    uniforms: Vec<u8>,
}

/// A pass between its beginning and `Present` or `EndPass`, and the name the program gave it.
struct OpenPass {
    pass: Pass,
    draws: Vec<PassItem>,
    label: Option<Arc<str>>,
    /// Built only when another pass asks to join; extended with each compatible join.
    uses: Option<HashMap<u32, bool>>,
    /// Whether its draws set a scissor or a viewport.
    rect_set: bool,
}

struct Screen {
    texture: wrela_gpu::Texture,
    view: wrela_gpu::TextureView,
    /// `rgba8unorm` (SDR), or `rgba16float` (HDR: linear, 1.0 SDR white).
    format: TextureFormat,
    width: u32,
    height: u32,
}

/// A buffer whose bytes for this submission are staged on the CPU and written to it at the
/// flush: the uniform ring, and the upload ring.
///
/// The upload ring holds this submission's buffer and texture writes: a copy of each from
/// there is recorded where the write is, so a write splits no work into submissions. Texture
/// rows are padded to 256 bytes, as buffer-to-texture copies require, and their uploads start
/// at that alignment too. (A queue write would have to come between two: the work recorded
/// before it must see the old
/// contents. Each submission waited for the GPU, which then started cold, and the pass after a
/// write measured a third more, the herd's culling 15 times.)
struct Ring {
    label: &'static str,
    buffer: wrela_gpu::Buffer,
    staging: Vec<u8>,
}

impl Ring {
    fn new(device: &wrela_gpu::Device, label: &'static str, size: u64) -> Ring {
        let buffer = device.create_buffer(&wrela_gpu::BufferDesc { label, size, mappable: false });
        Ring { label, buffer, staging: Vec::new() }
    }

    fn capacity(&self) -> u64 {
        self.buffer.size()
    }

    /// Replaces the buffer, after a flush (nothing is staged), with one that holds `bytes`:
    /// twice its size at least, a power of two. Whether it did: not where that's over `most`.
    fn grow_to_fit(&mut self, device: &wrela_gpu::Device, bytes: u64, most: u64) -> bool {
        debug_assert!(self.staging.is_empty());
        let size = bytes.max(self.capacity() * 2).next_power_of_two();
        if size > most {
            return false;
        }
        let staging = std::mem::take(&mut self.staging);
        *self = Ring { staging, ..Ring::new(device, self.label, size) };
        true
    }

    /// Writes this submission's staged bytes to the buffer.
    fn write(&mut self, device: &wrela_gpu::Device, counters: &mut Counters) {
        if !self.staging.is_empty() {
            device.write_buffer(&self.buffer, 0, &self.staging);
            counters.write_buffers += 1;
            counters.write_buffer_bytes += self.staging.len() as u64;
            self.staging.clear();
        }
    }
}

struct Timer {
    queries: wrela_gpu::QuerySet,
    /// How many queries `queries` holds.
    capacity: u32,
    /// Reused after each collection has finished reading its result.
    resolved: wrela_gpu::Buffer,
    readback: wrela_gpu::Buffer,
    /// What each pair of queries in this submission times.
    pending: Vec<Timed>,
    results: Vec<GpuTiming>,
    /// Whether every timing is kept for [`Gpu::take_timings`] (a run timed from its start): a
    /// timer the program started (`std::gpu::timed`) keeps only what answers its asks.
    keep: bool,
    /// The passes and dispatches timed so far: where an ask starts ([`Ask::from`]).
    noted: u64,
    /// Each pass and dispatch submitted and waited for alone, so none overlaps another: each
    /// one's own time (the serial mode).
    serial: bool,
    /// The submissions timed so far.
    submissions: usize,
}

/// A pass or dispatch timed in this submission, the frame and the call it's in, and its place
/// among those timed ([`Timer::noted`] before it).
struct Timed {
    frame: usize,
    call: usize,
    label: String,
    at: u64,
}

/// A program's settled read (`std::gpu::read_settled`): its copy, made after the work recorded
/// before it, and the submission it is in, once submitted.
struct Settling {
    request: u32,
    staging: wrela_gpu::Buffer,
    size: u64,
    done_at: Option<wrela_gpu::SubmissionIndex>,
    /// The device it was copied on was lost.
    lost: bool,
}

/// A program's ask for a label's time (`std::gpu::timed`).
struct Ask {
    request: u32,
    label: String,
    /// The passes and dispatches timed before it ([`Timer::noted`]): it times only its label's
    /// after it, in the order the program recorded them.
    from: u64,
    /// Its answer, once some of that work is read back: the first call that ran some, and the
    /// first start and the last end of its work in that call (nanoseconds, the device's clock).
    /// Passes and dispatches of one submission overlap on the GPU (on Apple GPUs a pass's start
    /// is near its submission's), so their spans can't be summed: the work took from the first
    /// one's start to the last one's end.
    time: Option<(usize, f64, f64)>,
}

/// What the executor did below the recorded stream, since its counters were last taken: the
/// physical half of a frame's draw count (G-draws), as the browser's `counters.json` has it. A
/// multi-draw is one `draw_calls` and its `count` `draws`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub compute_passes: u64,
    pub render_passes: u64,
    pub dispatches: u64,
    pub draws: u64,
    pub draw_calls: u64,
    pub write_buffers: u64,
    pub write_buffer_bytes: u64,
    pub write_textures: u64,
    pub bind_groups: u64,
    pub pipelines: u64,
}

impl Counters {
    fn add(&mut self, o: &Counters) {
        self.compute_passes += o.compute_passes;
        self.render_passes += o.render_passes;
        self.dispatches += o.dispatches;
        self.draws += o.draws;
        self.draw_calls += o.draw_calls;
        self.write_buffers += o.write_buffers;
        self.write_buffer_bytes += o.write_buffer_bytes;
        self.write_textures += o.write_textures;
        self.bind_groups += o.bind_groups;
        self.pipelines += o.pipelines;
    }

    /// Its fields by name, in order: what reports and tests list.
    pub fn fields(&self) -> [(&'static str, u64); 10] {
        [
            ("compute_passes", self.compute_passes),
            ("render_passes", self.render_passes),
            ("dispatches", self.dispatches),
            ("draws", self.draws),
            ("draw_calls", self.draw_calls),
            ("write_buffers", self.write_buffers),
            ("write_buffer_bytes", self.write_buffer_bytes),
            ("write_textures", self.write_textures),
            ("bind_groups", self.bind_groups),
            ("pipelines", self.pipelines),
        ]
    }
}

/// The display's guard. A run whose frames the GPU can't keep up with starves the display: on
/// 2026-11-13 and 2026-11-15 dispatches that ran for seconds, queued one behind another, made the
/// GPU restart and WindowServer's watchdog end every app on the machine (docs/dead-ends.md, "The
/// toolchain and the GPU"). So a run never has more than [`Behind::MOST`] frames' work
/// outstanding (it waits for the oldest), and it stops when a frame's work is still not done
/// [`Behind::limit`] after it was submitted. Neither changes a run the GPU keeps up with.
struct Behind {
    /// The frames submitted and not yet seen done, oldest first: which, the submission that
    /// ends it, and when.
    waiting: std::collections::VecDeque<(u64, wrela_gpu::SubmissionIndex, std::time::Instant)>,
    /// How many frames were submitted.
    count: u64,
    /// The longest a frame's work may be outstanding: `WRELA_GPU_BEHIND` seconds, else 1.5;
    /// `None` (0) for no limit.
    limit: Option<std::time::Duration>,
}

impl Behind {
    /// The most frames' work outstanding before a run waits for the oldest.
    const MOST: usize = 3;

    fn new() -> Self {
        let seconds = std::env::var("WRELA_GPU_BEHIND")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .unwrap_or(1.5);
        Self {
            waiting: std::collections::VecDeque::new(),
            count: 0,
            limit: (seconds > 0.0).then(|| std::time::Duration::from_secs_f64(seconds)),
        }
    }

    /// A frame's work was submitted to `device`: its last submission is the device's latest.
    fn submitted(&mut self, device: &wrela_gpu::Device) {
        if self.limit.is_none() {
            return;
        }
        self.count += 1;
        self.waiting.push_back((self.count, device.submitted(), std::time::Instant::now()));
    }

    /// Waits while more than [`Self::MOST`] frames are outstanding, and fails when the oldest
    /// has been outstanding longer than the limit.
    fn check(&mut self, device: &wrela_gpu::Device) -> Result<()> {
        let Some(limit) = self.limit else { return Ok(()) };
        loop {
            let done = device.completed();
            while self.waiting.front().is_some_and(|(_, s, _)| *s <= done) {
                self.waiting.pop_front();
            }
            let Some((frame, _, at)) = self.waiting.front().copied() else { return Ok(()) };
            let age = at.elapsed();
            if age > limit {
                return Err(Error::Gpu(format!(
                    "the GPU fell behind: frame {} of this run isn't done {:.1} s after it was \
                     submitted (the limit is {:.1} s, WRELA_GPU_BEHIND). The run stops here, \
                     before it starves the display. A dispatch or a pass is too heavy: cut it \
                     into parts spread over frames, or make it smaller",
                    frame - 1,
                    age.as_secs_f64(),
                    limit.as_secs_f64(),
                )));
            }
            if self.waiting.len() <= Self::MOST {
                return Ok(());
            }
            crate::cpu_time::blocked(|| std::thread::sleep(std::time::Duration::from_micros(200)));
        }
    }
}

/// The most work one submission may hold, by what bounds it: past either, what's recorded is
/// submitted first (`Gpu::room`). A submission is one job to the GPU's driver, and a driver
/// resets a GPU whose job runs too long (amdgpu: 10 s; a Metal command buffer of seconds hung
/// the M4's display twice): the step guard (`crate::step`) bounds each dispatch, these bound
/// the job. On 2026-11-19 a frame of 4,500 dispatches of a 512-step march (the engine's hole
/// audit, 640x360), each well within the guard, made one job of seconds on the Radeon once the
/// uniform ring stopped splitting frames (4f7fb1b), and amdgpu reset the GPU four times.
///
/// The guard's bounds of the dispatches queued, summed, as lines (`Gpu::line_ms`, the step's
/// limit, `WRELA_GPU_STEP` where it lowers it): two, what `Gpu::judge` holds a submission of
/// dispatches to.
const SUBMIT_LINES: f64 = 2.0;
/// The dispatches and passes queued: a pass has no bound the host knows, and by AGENTS.md's
/// rule each holds the GPU at most about a step (50 ms), so 64 of them about 3 s at most.
const SUBMIT_COMMANDS: u32 = 64;

/// How a buffer, texture or sampler was made: what `Gpu::recover` makes again at its handle.
#[derive(Clone, Copy, Debug)]
enum Made {
    Buffer {
        size: u32,
    },
    Texture {
        size: [u32; 3],
        format: TextureFormat,
        writable: bool,
        kind: TextureKind,
        levels: u32,
    },
    Sampler {
        linear: bool,
        repeat: bool,
        compare: Option<Compare>,
        mip_linear: bool,
        anisotropy: u32,
    },
}

/// Whether an error the GPU reported is the device lost (as either backend says it, or a
/// hang that ended its work).
fn says_lost(e: &Error) -> bool {
    let s = e.to_string();
    s.contains("lost the device") || s.contains("device was lost") || s.contains("Hang")
}

/// What the submission being recorded holds (`SUBMIT_LINES`, `SUBMIT_COMMANDS`).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Queued {
    commands: u32,
    bound_ms: f64,
    /// What the light passes in it took when last measured (ms): held under `LIGHT_JOB` of the
    /// line.
    passes_ms: f64,
}

impl Queued {
    /// Whether one more dispatch (its guard bound `ms`) or pass (0) would take it past either,
    /// its summed bounds past `most` ms: never one with nothing in it (a dispatch alone is the
    /// guard's to refuse).
    fn full(self, ms: f64, most: f64) -> bool {
        self.commands > 0 && (self.commands >= SUBMIT_COMMANDS || self.bound_ms + ms > most)
    }
}

/// A submission, timed on the GPU's clock once it's done (`Device::gpu_ms`), and the part of a
/// render pass it held, if any: what the host learns a pass's cost from and holds to the line.
struct Job {
    index: wrela_gpu::SubmissionIndex,
    /// The program's frame, and the host's count of frames ended (`Gpu::calls`) when it was
    /// made.
    frame: usize,
    call: usize,
    /// What of passes it holds: a piece alone (`alone`), or light passes among other work; and
    /// whether it holds dispatches.
    pieces: Vec<Piece>,
    alone: bool,
    dispatches: bool,
    /// A piece alone's own begin and end on the GPU's clock: its slot in `Gpu::stamps`.
    stamp: Option<u32>,
}

/// What of a pass's draws a submission holds: a pass that took an eighth of the line or less
/// when last measured (`LIGHT`) shares a submission with what's recorded around it; any other
/// is a submission of its own, or several, one a piece. (A sixteenth put the Radeon's `picture`
/// of SCENE 0, 3.2 ms, in one of its own every frame: +0.1 to 0.2 ms a frame.)
const LIGHT: f64 = 1.0 / 8.0;
/// How many pieces measured alone the guard's timestamps ring holds (each a pair of queries): a
/// slot is read once its submission is done, and given to no other piece before
/// (`Gpu::stamp_held`).
const STAMP_RING: u32 = 256;

/// The guard's timestamps of pieces measured alone: a ring of [`STAMP_RING`] query pairs, and
/// the buffers one pair is resolved into and read back through.
struct Stamps {
    queries: wrela_gpu::QuerySet,
    resolved: wrela_gpu::Buffer,
    readback: wrela_gpu::Buffer,
}

/// How many lines a piece alone may take, once, before the run ends: past it, at once (a real
/// runaway), within it, only when the next piece of its pass alone is over the line too.
const RECHECK_LINES: f64 = 4.0;
/// How many times its usual a pass's draws must take before the guard says so, naming the GPU's
/// idle before it.
const USUAL_TIMES: f64 = 3.0;
/// The flag's bit that says a lane ran out of fuel (`ir::fuel::FUEL_BIT`).
const IR_FUEL_BIT: u32 = 1 << 31;

/// How many destroyed handles the host remembers, to say when one a command names went.
const GONE_KEPT: usize = 1 << 16;

/// What the light passes of one submission may have taken, summed, as a share of the line.
const LIGHT_JOB: f64 = 1.0 / 4.0;
/// How many frames a light pass measured alone is drawn shared before a long submission sends it
/// to be measured alone again: a submission long for its dispatches would send it every other
/// frame (the Radeon's SCENE 0: +0.09 ms a frame).
const AGAIN_AFTER: usize = 32;

/// Draws `draws` (of `of`) of a pass labelled `label`.
#[derive(Clone, Debug)]
struct Piece {
    label: Arc<str>,
    draws: std::ops::Range<u32>,
    of: u32,
}

/// What a pass's draws cost, by its label: the most a draw took (a piece's time over its draws)
/// in the last frame measured and in the one being measured (ms), by the host's count of frames
/// (`Gpu::calls`), and the worst it has taken lately (falling a tenth a frame: a pass whose
/// draws are cheap some frames and heavy others, the engine's reference bands over sky and
/// over things, is costed by its heavy frames); whether it's to be measured alone again (it was
/// drawn as a light pass in a submission that took long), and when it last was; and whether a
/// piece of it alone was last measured over the line (`suspect`: the next over ends the run).
#[derive(Clone, Copy, Debug, Default)]
struct Cost {
    call: usize,
    last: f64,
    now: f64,
    worst: f64,
    again: bool,
    alone: usize,
    suspect: bool,
    /// The least a draw of it took measured alone and under the line (ms; 0 before one was):
    /// its usual, which a pass run at a lowered clock is told against.
    usual: f64,
}

impl Cost {
    fn per_draw(self) -> f64 {
        self.last.max(self.now).max(self.worst)
    }

    /// Moved on to frame `call` (`Gpu::calls`): the last frame's worst kept, falling.
    fn at(&mut self, call: usize) {
        if call > self.call {
            let fall = 0.9f64.powi((call - self.call).min(64) as i32);
            *self = Cost {
                call,
                last: self.now,
                now: 0.0,
                worst: self.worst.max(self.now) * fall,
                ..*self
            };
        }
    }
}

/// A split forced on render passes by `WRELA_GPU_SPLIT`: `<k>` cuts every pass of two draws or
/// more into `k` pieces (as many as it has draws, at most), `<label>:<k>` only the passes so
/// labelled. What a test of the split (the picture the same to the bit) and a measure of its
/// cost (each piece loads and stores its targets) set; a run never needs it.
fn forced_split() -> Option<(Option<String>, u32)> {
    let v = std::env::var("WRELA_GPU_SPLIT").ok()?;
    let (label, k) = match v.rsplit_once(':') {
        Some((l, k)) => (Some(l.to_string()), k),
        None => (None, v.as_str()),
    };
    let k: u32 = k.trim().parse().ok()?;
    (k >= 2).then_some((label, k))
}

/// The pieces of a pass of `draws` draws whose cost is `cost` (none known: each draw alone),
/// each at most `room` ms by it; a pass of no draws is one piece.
fn pieces(draws: u32, cost: Option<Cost>, room: f64) -> Vec<std::ops::Range<u32>> {
    let size = match cost {
        None => 1,
        Some(c) if c.per_draw() <= 0.0 => draws.max(1),
        Some(c) => ((room / c.per_draw()).floor() as u32).clamp(1, draws.max(1)),
    };
    if draws == 0 {
        return vec![0..0];
    }
    (0..draws.div_ceil(size)).map(|k| k * size..((k + 1) * size).min(draws)).collect()
}

/// The invocations a dispatch of `p` over `groups` runs: an indirect dispatch's, its stated most
/// groups' (`most:`), else unknown (the GPU's to hold).
fn invocations(p: &Pipeline, groups: Result<[u32; 3], (u32, u32, u32)>) -> Option<u64> {
    match groups {
        Ok([x, y, z]) => Some(u64::from(x) * u64::from(y) * u64::from(z) * p.per_group),
        Err((_, _, stated)) if stated > 0 => Some(u64::from(stated) * p.per_group),
        Err(_) => None,
    }
}

pub(crate) struct Gpu {
    counters: Counters,
    /// The longest any submission holding a pass may take on the GPU (ms): `crate::step`'s
    /// limit, AGENTS.md's 50 ms. A pass is cut into pieces of at most a quarter of it by what its
    /// draws took before; a piece measured over it ends the run.
    line_ms: f64,
    /// What each label's draws cost, learned from the pieces measured.
    costs: HashMap<Arc<str>, Cost>,
    /// A split forced on passes (`WRELA_GPU_SPLIT`, for tests and measures): the label it's
    /// forced on (every label, `None`) and how many pieces.
    forced: Option<(Option<String>, u32)>,
    /// The submissions not yet measured, oldest first.
    jobs: std::collections::VecDeque<Job>,
    /// The piece the next submission holds alone, or the light passes it holds.
    job_piece: Option<Piece>,
    job_light: Vec<Piece>,
    /// Timestamps of pieces measured alone (where the device has them and no timer stamps
    /// passes): a ring of query pairs, the next slot, and the slot the next submission's piece
    /// holds. The guard judges a piece over the line by its pass's own time from these.
    stamps: Option<Stamps>,
    stamp_next: u32,
    /// Which slots a piece not yet measured holds: never given another before it's read.
    stamp_held: Vec<bool>,
    job_stamp: Option<u32>,
    /// Every measured job's piece and time, for tests.
    #[cfg(test)]
    measured: Vec<(Option<Piece>, f64)>,
    /// What the submission being recorded holds, and how many submissions have been made.
    queued: Queued,
    submissions: u64,
    /// The most any submission has held: its commands, and (apart) its summed bound.
    largest: Queued,
    /// What it was made with, kept to make it again on a new device when this one is lost
    /// (`Gpu::recover`): the build, how it times, and each live resource's making.
    manifest: Manifest,
    shaders: Vec<Vec<u8>>,
    timing: Timing,
    made: HashMap<u32, Made>,
    device: wrela_gpu::Device,
    limits: wrela_gpu::Limits,
    align: u64,
    pipelines: Vec<Pipeline>,
    /// Each pipeline's key (`pipeline_key`): a hot reload keeps a pipeline whose manifest entry
    /// and WGSL are the same.
    pipeline_keys: Vec<u64>,
    /// Pipelines no running build uses, by key, kept for the next: those another build was
    /// got ready with (`std::reload::prepare`; `true`), kept until it starts or is let go, and
    /// the last build's that the running one doesn't use (`false`), kept until the next swap.
    spare: HashMap<u64, (Pipeline, bool)>,
    /// How many pipelines it has built, here or on a [`Builder`]'s thread: for a test or a
    /// timing of what a swap built.
    built: Arc<AtomicUsize>,
    /// The wall time it has spent building the builds' pipelines it took (ms), all threads at
    /// once: a load's pipeline creation, as `--cpu` reports it.
    build_ms: f64,
    /// Whether each pipeline it builds is reported on stderr ([`crate::pipeline_report`]).
    report: bool,
    resources: HashMap<u32, Resource>,
    /// Each pipeline's cached bind groups, by the bindings they hold.
    bind_groups: Vec<HashMap<Arc<[Binding]>, wrela_gpu::BindGroup>>,
    /// For each resource, the cached bind groups it's in: destroying it drops them.
    bound_in: HashMap<u32, HashSet<BindGroupKey>>,
    /// Handles destroyed, with the frame and the call they were destroyed in, so a command that
    /// names one after says when it went (`no_buffer`); the oldest forgotten past `GONE_KEPT`.
    gone: HashMap<u32, (usize, usize)>,
    /// The uniform ring, bound with dynamic offsets, and the upload ring.
    ring: Ring,
    upload: Ring,
    encoder: Option<wrela_gpu::CommandEncoder>,
    pass: Option<OpenPass>,
    /// A pass that ended, not yet recorded: the next pass continues it if it begins on the same
    /// targets, keeps them (`Pass::joins`) and has the same label, with no `Label` between.
    ended: Option<OpenPass>,
    screen: Option<Screen>,
    /// The screen's format, and that of the screen made next ([`Gpu::show_screen`]):
    /// [`SCREEN_FORMAT`], or `rgba16float` once the program asks HDR and is let have it.
    screen_format: TextureFormat,
    /// Reused between buffer reads: mapping is finished before the next copy uses it. A build
    /// that retires lets it go.
    buffer_readback: Option<wrela_gpu::Buffer>,
    /// Reused between captures of the screen and of passes' targets: mapping is finished before
    /// the next copy uses it. A build that retires lets it go.
    image_readback: Option<wrela_gpu::Buffer>,
    /// Each label's pass's target, the last pass of that label that drew into a texture: its
    /// colour, else its depth, as (texture, level, layer) ([`Gpu::read_target`], a test's
    /// golden of a pass).
    labelled: HashMap<Arc<str>, (u32, u32, u32)>,
    /// The labels in `labelled`, in the order their passes last drew: a comparison's order.
    labelled_order: Vec<Arc<str>>,
    /// The program's asks for a label's time (`std::gpu::timed`) not yet answered.
    asked: Vec<Ask>,
    /// The program's settled reads (`std::gpu::read_settled`) not yet answered, in the order
    /// asked.
    settling: Vec<Settling>,
    /// The calls ended so far: a timing's call, and an ask's.
    calls: usize,
    /// The display's guard: how far behind the GPU is on this run's frames.
    behind: Behind,
    /// The GPU's check of indirect dispatches' groups (`crate::step`).
    indirect: crate::step::Indirect,
    timer: Option<Timer>,
    /// The name the program gave the passes and dispatches after it in this frame (`Label`).
    label: Option<Arc<str>>,
    /// Whether that name was given since the last pass or dispatch: a screen pass takes it then,
    /// and else ends it (named `screen pass`), so its time is never summed into the label of
    /// the work before it.
    label_fresh: bool,
    /// The label of the timed compute run the encoder has open (`CommandEncoder::
    /// timed_compute_open`): a timed span's dispatches of one label in a row are timed as one
    /// run, with no encoder of their own each.
    timed_run: Option<Arc<str>>,
    frame: usize,
    /// A debug build's: the flag its pipelines' bounds checks set (the manifest's
    /// `debug_flag`), read after each frame.
    debug_flag: Option<DebugFlag>,
    /// The limits the program sees (`wrela.limit`), with its optional features
    /// ([`feature_bits`]) and, with subgroups, their sizes.
    program_limits: wrela_abi::Limits,
    /// The window the screen is shown in, when there is one ([`Gpu::attach_window`]).
    presenter: Option<Presenter>,
    /// How the program last asked to show the screen (`ConfigureScreen`), done between frames.
    screen_asked: Option<(bool, bool)>,
    /// Whether an offscreen screen goes HDR when asked (a test's `hdr=1`): else it stays SDR.
    offscreen_hdr: bool,
    /// Keeps this device clocked while the program works on the CPU (`--keep-clocked`).
    warmer: Option<Warmer>,
    /// The frame at whose end the device is reported lost (`WRELA_GPU_LOSE_AT`), if one is
    /// given: how a lost device's handling is tested on a machine whose GPU can't be made to
    /// lose it safely.
    lose_at: Option<usize>,
}

/// The flag a debug build's bounds checks set, and the buffer it's read back through, made
/// once.
struct DebugFlag {
    buffer: wrela_gpu::Buffer,
    /// Copies of the flag, one a frame, `FLAG_RING` of them: a frame's is read once its work is
    /// done, so no frame waits for the GPU to read it.
    ring: wrela_gpu::Buffer,
    next: u64,
    /// The copy recorded in the frame being submitted: its slot, and the draws whose arguments
    /// it holds.
    queued: Option<(u64, Vec<Drawn>)>,
    /// The copies submitted and not yet read: the submission that holds each, its slot, its
    /// frame and the draws whose arguments it holds.
    pending: std::collections::VecDeque<(wrela_gpu::SubmissionIndex, u64, usize, Vec<Drawn>)>,
    /// The name each pipeline's last pass or dispatch had (a fault names it).
    labels: HashMap<u32, Arc<str>>,
    /// The buffer's words: 1, the flag; in a counting build, then each pipeline's tally
    /// (`Manifest::tally_words`), copied with the flag each frame and zeroed after the copy
    /// from `zeros`, so each frame's copy holds its own.
    words: u64,
    zeros: Option<wrela_gpu::Buffer>,
    /// The words of the flag and the tallies: in a counting build, the rest of the buffer
    /// holds, after them, the arguments of each indirect draw of the frame
    /// ([`DRAWN_WORDS`] each, at most [`DRAWN_MOST`]), copied after its pass.
    tally: u64,
    /// This frame's indirect draws whose arguments are copied, in order.
    drawn: Vec<Drawn>,
}

/// An indirect draw whose arguments a counting run reads back: its pipeline, its pass's label
/// and whether it's indexed (five words of arguments, else four).
type Drawn = (u32, Arc<str>, bool);

/// The most indirect draws a frame of a counting run reads the arguments of, and the words each
/// takes (an indexed draw's five).
const DRAWN_MOST: u64 = 4096;
const DRAWN_WORDS: u64 = 5;

/// How many frames' copies of the flag are kept ([`DebugFlag::ring`]): more than are ever in
/// flight.
const FLAG_RING: u64 = 64;

/// The limits a program sees, from the device's.
pub(crate) fn limits_of(l: &wrela_gpu::Limits) -> wrela_abi::Limits {
    let clamp = |x: u64| u32::try_from(x).unwrap_or(u32::MAX);
    wrela_abi::Limits {
        max_texture_size: l.max_texture_dimension_2d,
        max_buffer_size: clamp(l.max_buffer_size.min(l.max_storage_buffer_binding_size)),
        max_storage_buffers_per_stage: l.max_storage_buffers_per_shader_stage,
        max_uniform_buffer_binding_size: clamp(l.max_uniform_buffer_binding_size),
        max_workgroup_storage_size: l.max_compute_workgroup_storage_size,
        max_workgroup_invocations: l.max_compute_invocations_per_workgroup,
        max_workgroup_size: l.max_compute_workgroup_size,
        max_workgroups_per_dimension: l.max_compute_workgroups_per_dimension,
        ..wrela_abi::Limits::HERE
    }
}

/// Waits for the device to finish its work, counted as time blocked on the GPU
/// ([`crate::cpu_time`]).
fn wait(device: &wrela_gpu::Device) {
    crate::cpu_time::blocked(|| device.wait(device.submitted()));
}

/// Opens the machine's GPU (`wrela_gpu`: Metal, or Vulkan) with its optional features (those
/// `keep` names, or all), as the browser runtime asks for them: the program sees the GPU's real
/// limits (`wrela.limit`). A hardware GPU (a discrete one before an integrated one), never a
/// software one (Mesa's llvmpipe is a Vulkan device too) unless `WRELA_ALLOW_SOFTWARE` says so.
pub fn open_device(label: &str, keep: Option<Vec<String>>) -> Result<wrela_gpu::Device> {
    let device = wrela_gpu::Device::open(&wrela_gpu::DeviceOptions {
        label: label.to_string(),
        allow_software: crate::software_allowed(),
        features: keep,
    })
    .map_err(|e| Error::Gpu(e.0))?;
    let info = device.info();
    admits(&info.name, info.device_type, crate::software_allowed())?;
    Ok(device)
}

/// `wrela-host --adapter`: the machine's GPU as `wrela_gpu` opens it: its name, driver, features
/// (one a line, WebGPU's names) and limits (one a line).
pub fn adapter_report() -> Result<String> {
    use std::fmt::Write as _;
    let device = open_device("adapter", None)?;
    let info = device.info();
    let mut s = String::new();
    let _ = writeln!(s, "adapter {} ({:?}, {:?})", info.name, info.backend, info.device_type);
    let _ = writeln!(s, "driver {} {}", info.driver, info.driver_info);
    let _ = writeln!(s, "subgroups {} to {}", info.subgroup_min_size, info.subgroup_max_size);
    s.push_str(&device.report());
    let _ = writeln!(s, "\nfeatures");
    for name in &device.features().0 {
        let _ = writeln!(s, "  {name}");
    }
    let _ = writeln!(s, "\nlimits");
    let limits = format!("{:#?}", device.limits());
    for line in limits.lines().filter(|l| l.starts_with("    ")) {
        let _ = writeln!(s, "  {}", line.trim().trim_end_matches(','));
    }
    Ok(s)
}

/// Whether a device of `kind` may run a build: a hardware GPU always, a software one (`Cpu`)
/// only where `software` allows it, and a virtual or unknown one never.
fn admits(name: &str, kind: wrela_gpu::DeviceType, software: bool) -> Result<()> {
    match kind {
        wrela_gpu::DeviceType::IntegratedGpu | wrela_gpu::DeviceType::DiscreteGpu => Ok(()),
        wrela_gpu::DeviceType::Cpu if software => Ok(()),
        wrela_gpu::DeviceType::Cpu => Err(Error::Gpu(format!(
            "hardware GPU required; {name} reports {kind:?} (set {}=1 to run on it: its timings are a CPU's)",
            crate::ALLOW_SOFTWARE
        ))),
        _ => Err(Error::Gpu(format!("hardware GPU required; {name} reports {kind:?}"))),
    }
}

impl Gpu {
    /// Opens the GPU and builds every pipeline. `shaders[i]` is pipeline `i`'s WGSL source.
    /// `timing` says whether passes are timed, and how; `report`, whether each pipeline built
    /// is reported on stderr ([`crate::pipeline_report`]).
    pub(crate) fn new(
        manifest: &Manifest,
        shaders: &[Vec<u8>],
        timing: Timing,
        report: bool,
    ) -> Result<Gpu> {
        let timestamps = timing != Timing::Off;
        // Shaders are handed over as the target's own (`passthrough`): a build for another
        // machine can't run here.
        if manifest.target != wrela_abi::Target::HERE {
            return Err(Error::Gpu(format!(
                "this is a build for {}, and this machine runs {} (wrela build --target {})",
                manifest.target.name(),
                wrela_abi::Target::HERE.name(),
                wrela_abi::Target::HERE.name()
            )));
        }
        // Every optional feature wherever the device has it (timestamps too: a program may ask
        // for its work's time). `WRELA_FEATURES` (`none`, or a comma list of the ones to keep)
        // masks them, as test mode's `features=` does in a browser: the device is opened without
        // the masked ones, so its pipelines and commands are a device's without them, though the
        // GPU has them. Timestamps stay, which a profile's timing needs.
        let keep: Option<Vec<String>> = std::env::var("WRELA_FEATURES").ok().map(|v| {
            if v == "none" {
                Vec::new()
            } else {
                v.split(',').map(|f| f.trim().to_string()).collect()
            }
        });
        let kept = |name: &str| keep.as_ref().is_none_or(|k| k.iter().any(|x| x == name));
        let asked: Vec<String> = wrela_abi::FEATURES
            .iter()
            .filter(|f| **f == "timestamp-query" || **f == "indirect-first-instance" || kept(f))
            .map(|f| f.to_string())
            .collect();
        let device = open_device("wrela-host", Some(asked))?;
        // Indirect arguments include first_instance, including arguments written by a kernel.
        // Without this feature, a non-zero value can silently discard the draw.
        for need in
            ["indirect-first-instance"].into_iter().chain(timestamps.then_some("timestamp-query"))
        {
            if !device.features().has(need) {
                return Err(Error::Gpu(format!("{} doesn't support {need}", device.info().name)));
            }
        }
        let indirect = crate::step::Indirect::new(&device)?;
        // Pipelines are compiled for their target's limits: a device with less of any can't run
        // them.
        let need = &manifest.target.limits().device;
        if let Some(short) = limits_of(device.limits()).short_of(need) {
            return Err(Error::Gpu(format!(
                "the device's {short} is less than the machine's that pipelines are compiled for ({})",
                device.info().name
            )));
        }

        // The guard's unit on this device (not in a test: each test opens its own device).
        if !cfg!(test) {
            ensure_unit(&device.info())?;
        }
        let limits = device.limits().clone();
        let mut program_limits = limits_of(&limits);
        program_limits.features = feature_bits(device.features(), kept);
        program_limits.step_work = crate::step::step_work(wrela_abi::Target::HERE);
        program_limits.lane_work = wrela_abi::lane_work(
            program_limits.step_work,
            crate::step::width(wrela_abi::Target::HERE) as u32,
        );
        if program_limits.has("subgroups") {
            let info = device.info();
            program_limits.subgroup_min_size = info.subgroup_min_size;
            program_limits.subgroup_max_size = info.subgroup_max_size;
        }
        let align = u64::from(
            limits
                .min_uniform_buffer_offset_alignment
                .max(limits.min_storage_buffer_offset_alignment),
        );
        let ring = Ring::new(&device, "uniform ring", RING_START);
        let upload = Ring::new(&device, "upload ring", u64::from(stream::UPLOAD_START));
        let timer = timestamps.then(|| Timer::new(&device, timing == Timing::Serial, true));
        let mut gpu = Gpu {
            counters: Counters::default(),
            line_ms: crate::step::limit_ms(),
            forced: forced_split(),
            costs: HashMap::new(),
            jobs: std::collections::VecDeque::new(),
            job_piece: None,
            job_light: Vec::new(),
            stamps: if device.features().has("timestamp-query") {
                device.create_query_set(2 * STAMP_RING).ok().map(|queries| {
                    let (resolved, readback) = query_buffers(&device, 2);
                    Stamps { queries, resolved, readback }
                })
            } else {
                None
            },
            stamp_next: 0,
            stamp_held: vec![false; STAMP_RING as usize],
            job_stamp: None,
            #[cfg(test)]
            measured: Vec::new(),
            queued: Queued::default(),
            submissions: 0,
            largest: Queued::default(),
            manifest: manifest.clone(),
            shaders: shaders.to_vec(),
            timing,
            made: HashMap::new(),
            device,
            limits,
            align,
            pipelines: Vec::new(),
            pipeline_keys: Vec::new(),
            spare: HashMap::new(),
            built: Arc::new(AtomicUsize::new(0)),
            build_ms: 0.0,
            report,
            resources: HashMap::new(),
            bind_groups: vec![HashMap::new(); manifest.pipelines.len()],
            bound_in: HashMap::new(),
            gone: HashMap::new(),
            ring,
            upload,
            encoder: None,
            pass: None,
            ended: None,
            screen: None,
            screen_format: SCREEN_FORMAT,
            buffer_readback: None,
            image_readback: None,
            labelled: HashMap::new(),
            labelled_order: Vec::new(),
            asked: Vec::new(),
            settling: Vec::new(),
            calls: 0,
            behind: Behind::new(),
            indirect,
            timer,
            label: None,
            label_fresh: false,
            timed_run: None,
            frame: 0,
            debug_flag: None,
            program_limits,
            presenter: None,
            screen_asked: None,
            offscreen_hdr: false,
            warmer: None,
            lose_at: std::env::var("WRELA_GPU_LOSE_AT").ok().and_then(|v| v.trim().parse().ok()),
        };
        gpu.take_pipelines(manifest, shaders)?;
        gpu.warmer = Warmer::for_device(&gpu.device);
        Ok(gpu)
    }

    /// Builds the manifest's pipelines, keeping each of the ones it has whose manifest entry
    /// and WGSL are the same.
    fn take_pipelines(&mut self, manifest: &Manifest, shaders: &[Vec<u8>]) -> Result<()> {
        if self.debug_flag.is_none() && manifest.pipelines.iter().any(|p| p.debug_flag.is_some()) {
            let tally = u64::from(manifest.tally_words.max(1));
            let words = if tally > 1 { tally + DRAWN_MOST * DRAWN_WORDS } else { tally };
            self.debug_flag = Some(DebugFlag {
                buffer: self.device.create_buffer(&wrela_gpu::BufferDesc {
                    label: "debug flag",
                    size: 4 * words,
                    mappable: false,
                }),
                ring: readback_buffer(&self.device, 4 * FLAG_RING * words),
                next: 0,
                queued: None,
                pending: std::collections::VecDeque::new(),
                labels: HashMap::new(),
                words,
                zeros: (words > 1).then(|| {
                    self.device.create_buffer(&wrela_gpu::BufferDesc {
                        label: "tally zeros",
                        size: 4 * (words - 1),
                        mappable: false,
                    })
                }),
                tally,
                drawn: Vec::new(),
            });
        }
        let old_keys = std::mem::take(&mut self.pipeline_keys);
        let mut old: HashMap<u64, Pipeline> =
            old_keys.into_iter().zip(std::mem::take(&mut self.pipelines)).collect();
        // The last build's spares go; those got ready for a build stay.
        self.spare.retain(|_, (_, prepared)| *prepared);
        let mut kept = Vec::new();
        let mut missing = Vec::new();
        for (p, source) in manifest.pipelines.iter().zip(shaders) {
            let key = pipeline_key(p, source);
            let have = old.remove(&key).or_else(|| self.spare.remove(&key).map(|s| s.0));
            if have.is_none() {
                missing.push((p, source.as_slice()));
            }
            kept.push(have);
            self.pipeline_keys.push(key);
        }
        let began = std::time::Instant::now();
        let mut built = self.builder().build_many(&missing)?.into_iter();
        self.build_ms += began.elapsed().as_secs_f64() * 1000.0;
        for have in kept {
            self.pipelines
                .push(have.unwrap_or_else(|| built.next().expect("one built for each missing")));
        }
        for (key, pipeline) in old {
            self.spare.entry(key).or_insert((pipeline, false));
        }
        self.bind_groups = vec![HashMap::new(); manifest.pipelines.len()];
        // Built for the screen as it is (a build taken while it's HDR draws it HDR).
        self.screen_variants()?;
        self.check_errors()
    }

    /// A fresh program on the same device: only immutable pipelines and upload allocations
    /// survive. In particular, a program that doesn't draw mustn't see the last one's screen.
    pub(crate) fn restart_build(&mut self, manifest: &Manifest, shaders: &[Vec<u8>]) -> Result<()> {
        self.finish_build()?;
        // The screen as a new program's is: SDR, none made, and so the program is told; a window
        // shown HDR is set back at the frame's end unless the program asks it again.
        self.screen = None;
        self.screen_format = SCREEN_FORMAT;
        self.program_limits.screen = 0;
        self.program_limits.headroom = wrela_abi::Limits::HERE.headroom;
        self.screen_asked = Some((false, false));
        self.frame = 0;
        self.spare.clear();
        // A timer the host runs from the start begins again; one the last program asked for
        // goes with it.
        self.timer = match self.timer.take() {
            Some(t) if t.keep => Some(Timer::new(&self.device, t.serial, true)),
            _ => None,
        };
        self.take_build(manifest, shaders)
    }

    /// Takes the build `manifest` and `shaders` in place of the one whose work
    /// [`Gpu::finish_build`] finished: its pipelines, keeping each of the ones it has that are
    /// the same.
    fn take_build(&mut self, manifest: &Manifest, shaders: &[Vec<u8>]) -> Result<()> {
        self.manifest = manifest.clone();
        self.shaders = shaders.to_vec();
        self.take_pipelines(manifest, shaders)
    }

    /// Finishes the retiring build's commands before releasing the resources they used.
    fn finish_build(&mut self) -> Result<()> {
        self.pass = None;
        self.label = None;
        self.flush()?;
        self.collect()?;
        wait(&self.device);
        self.resources.clear();
        self.bound_in.clear();
        // What it made and destroyed goes too: a device made again later makes only the next
        // build's resources (`Gpu::recover`), and its handles name that build's.
        self.made.clear();
        self.gone.clear();
        self.debug_flag = None;
        // The retiring program's asks for a time and its labels' targets go with it: a later
        // build's requests reuse its numbers, and its handles name other resources.
        self.asked.clear();
        self.settling.clear();
        self.labelled.clear();
        self.labelled_order.clear();
        // Its largest read sized these; the next build makes its own as it reads.
        self.buffer_readback = None;
        self.image_readback = None;
        Ok(())
    }

    /// What builds pipelines on this device, on any thread.
    pub(crate) fn builder(&self) -> Builder {
        Builder {
            device: self.device.clone(),
            limits: self.limits.clone(),
            built: self.built.clone(),
            report: self.report,
            epoch: self.program_limits.epoch,
        }
    }

    /// Keeps `ready`, pipelines got ready for another build (`std::reload::prepare`), until a
    /// swap takes them or [`Gpu::let_go`] drops them; `keys` are all that build's pipelines',
    /// so the spares it would use are kept as long. One the running build has is dropped, and
    /// so is every one built on a device lost since (`Gpu::recover`): a swap builds those again.
    pub(crate) fn keep_ready(&mut self, ready: Ready, keys: &[u64]) {
        let pipelines =
            if ready.epoch == self.program_limits.epoch { ready.pipelines } else { Vec::new() };
        for (key, pipeline) in pipelines {
            if !self.pipeline_keys.contains(&key) {
                self.spare.insert(key, (pipeline, true));
            }
        }
        for key in keys {
            if let Some(s) = self.spare.get_mut(key) {
                s.1 = true;
            }
        }
    }

    /// The keys of the pipelines it holds: the running build's and the spares.
    pub(crate) fn pipeline_keys(&self) -> HashSet<u64> {
        self.pipeline_keys.iter().chain(self.spare.keys()).copied().collect()
    }

    /// Drops the spare pipelines with these keys: those got ready for a build that's let go.
    pub(crate) fn let_go(&mut self, keys: &[u64]) {
        for key in keys {
            self.spare.remove(key);
        }
    }

    /// How many pipelines it has built since it was opened.
    pub(crate) fn pipelines_built(&self) -> usize {
        self.built.load(Ordering::Relaxed)
    }

    /// The wall time spent building the pipelines builds took, in milliseconds.
    pub(crate) fn pipelines_build_ms(&self) -> f64 {
        self.build_ms
    }
}

/// Pipelines built for another build, by key (`std::reload::prepare`), and the device epoch
/// they were built on (`std::gpu::device_epoch`): another epoch's are another device's.
pub(crate) struct Ready {
    epoch: u32,
    pipelines: Vec<(u64, Pipeline)>,
}

/// Builds pipelines on a device: [`Gpu::builder`]'s, on a thread of its own while a program
/// runs (`std::reload::prepare`).
pub(crate) struct Builder {
    device: wrela_gpu::Device,
    limits: wrela_gpu::Limits,
    built: Arc<AtomicUsize>,
    /// Whether each pipeline built is reported on stderr ([`crate::pipeline_report`]).
    report: bool,
    /// The device epoch its device is ([`Ready::epoch`]).
    epoch: u32,
}

impl Builder {
    /// Each of the build's pipelines that `have` doesn't hold (by key), built: with its key.
    pub(crate) fn build_all(
        &self,
        manifest: &Manifest,
        shaders: &[Vec<u8>],
        have: &HashSet<u64>,
    ) -> Result<Ready> {
        let mut keys = Vec::new();
        let mut missing = Vec::new();
        for (p, source) in manifest.pipelines.iter().zip(shaders) {
            let key = pipeline_key(p, source);
            if !have.contains(&key) && !keys.contains(&key) {
                keys.push(key);
                missing.push((p, source.as_slice()));
            }
        }
        let pipelines = keys.into_iter().zip(self.build_many(&missing)?).collect();
        Ok(Ready { epoch: self.epoch, pipelines })
    }

    /// Each of `jobs` built, in order, on as many threads as the machine has cores. Returns the
    /// first failure in order, if any.
    fn build_many(
        &self,
        jobs: &[(&wrela_abi::manifest::Pipeline, &[u8])],
    ) -> Result<Vec<Pipeline>> {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        // A report times each pipeline's build, so they're built one at a time.
        let threads = if self.report { 1 } else { cores.min(jobs.len()) };
        if threads <= 1 {
            return jobs.iter().map(|(p, source)| self.build(p, source)).collect();
        }
        let next = AtomicUsize::new(0);
        let mut results: Vec<Option<Result<Pipeline>>> = (0..jobs.len()).map(|_| None).collect();
        let done = Mutex::new(&mut results);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some((p, source)) = jobs.get(i) else { break };
                        let r = self.build(p, source);
                        crate::locked(&done)[i] = Some(r);
                    }
                });
            }
        });
        results.into_iter().map(|r| r.expect("every job ran")).collect()
    }

    fn build(&self, p: &wrela_abi::manifest::Pipeline, source: &[u8]) -> Result<Pipeline> {
        self.built.fetch_add(1, Ordering::Relaxed);
        let began = std::time::Instant::now();
        let shader_err = |message: String| Error::Shader {
            pipeline: p.name.clone(),
            shader: p.shader.clone(),
            message,
        };
        if let Some(u) = &p.uniform {
            let limit = match u.space {
                UniformSpace::Uniform => self.limits.max_uniform_buffer_binding_size,
                UniformSpace::Storage => self.limits.max_storage_buffer_binding_size,
            };
            if u64::from(u.size) > limit {
                return Err(shader_err(format!(
                    "its uniform block is {} bytes; the limit is {limit}",
                    u.size
                )));
            }
        }
        // A shader that needs a feature this device hasn't isn't built: the checker refuses a
        // command that records it, naming the feature.
        let features = self.device.features();
        let unavailable = p.requires.iter().any(|f| !features.has(f));
        let layout = self.device.create_bind_group_layout(&layout_entries(p));
        let kind = if unavailable {
            Kind::Unavailable
        } else {
            // Its MSL or SPIR-V, as the compiler wrote it for this machine's API.
            let module =
                self.device.create_shader_module(&p.shader, source).map_err(|e| shader_err(e.0))?;
            match &p.stage {
                Stage::Compute { entry, workgroup_size } => Kind::Compute(
                    self.device
                        .create_compute_pipeline(&wrela_gpu::ComputePipelineDesc {
                            label: &p.name,
                            layout: &layout,
                            module: &module,
                            entry,
                            workgroup_size: *workgroup_size,
                        })
                        .map_err(|e| shader_err(e.0))?,
                ),
                Stage::Render {
                    vertex_entry,
                    fragment_entry,
                    blends,
                    cull,
                    depth_bias,
                    depth,
                    targets,
                    ..
                } => {
                    let mut kind = Kind::Render {
                        module,
                        vertex: vertex_entry.clone(),
                        fragment: fragment_entry.clone(),
                        blends: blends.clone(),
                        cull: *cull,
                        depth_bias: *depth_bias,
                        depth: *depth,
                        variants: HashMap::new(),
                    };
                    // A variant for each of its targets (the passes the program draws it in, which
                    // the compiler knows), now: a shader's errors come at load, and no pass waits to
                    // compile one.
                    for t in targets {
                        // A target the device can't draw into (rg11b10ufloat without its feature)
                        // is one the checker refuses a pass of.
                        if t.colors().any(|c| !c.info().renderable.on(|f| features.has(f))) {
                            continue;
                        }
                        // Nor one it can't blend into as the pipeline blends (a 32-bit float
                        // target without float32-blendable): the checker refuses such a draw.
                        if t.colors().enumerate().any(|(k, c)| {
                            blends.get(k).is_some_and(|&b| b != Blend::Replace)
                                && !c.info().blendable.on(|f| features.has(f))
                        }) {
                            continue;
                        }
                        let depth = t.depth.then_some(TextureFormat::Depth32Float);
                        let colors = t.colors().collect();
                        render_variant(&self.device, &p.name, &layout, &mut kind, (colors, depth))
                            .map_err(shader_err)?;
                    }
                    kind
                }
            }
        };
        if self.report {
            self.report(p, source, &kind, began.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(Pipeline {
            name: p.name.as_str().into(),
            kind,
            layout,
            uniform: p.uniform.as_ref().map(|u| (u.binding, u.size)),
            bindings: p.bindings.clone(),
            debug_flag: p.debug_flag,
            sizes: p.sizes.clone(),
            bound: crate::step::Bound::of(p),
            screen_depths: match &p.stage {
                Stage::Render { targets, .. } => {
                    targets.iter().filter(|t| t.screen).map(|t| t.depth).collect()
                }
                Stage::Compute { .. } => Vec::new(),
            },
            per_group: match &p.stage {
                Stage::Compute { workgroup_size, .. } => {
                    workgroup_size.iter().map(|&n| u64::from(n)).product()
                }
                Stage::Render { .. } => 1,
            },
        })
    }
}

/// Bind group 0's layout of pipeline `p`, as the compiler wrote the shader's argument tables for
/// (`wrela_abi::metal`): the same bindings, each seen by the same stages.
pub(crate) fn layout_entries(p: &wrela_abi::manifest::Pipeline) -> Vec<wrela_gpu::LayoutEntry> {
    wrela_abi::metal::layout(p)
        .iter()
        .map(|e| wrela_gpu::LayoutEntry {
            binding: e.binding,
            visibility: e.visibility,
            ty: binding_type(e, p),
            index_at: if e.table { wrela_abi::table::index_offset(p, e.binding) } else { None },
        })
        .collect()
}

/// The uniform buffer a bind group's shader reads its storage buffers' bound sizes from
/// (`wrela_abi::metal`): `words`, one a buffer in `BufferSizes::buffers`' order, padded to 16
/// bytes. Staged: it lands before the submission that first binds it.
pub(crate) fn sizes_buffer(
    device: &wrela_gpu::Device,
    label: &str,
    words: impl IntoIterator<Item = u32>,
) -> wrela_gpu::Buffer {
    let mut bytes: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    bytes.resize(bytes.len().next_multiple_of(16).max(16), 0);
    let buffer = device.create_buffer(&wrela_gpu::BufferDesc {
        label,
        size: bytes.len() as u64,
        mappable: false,
    });
    device.write_buffer(&buffer, 0, &bytes);
    buffer
}

/// What a layout entry of pipeline `p` binds.
fn binding_type(
    e: &wrela_abi::metal::LayoutEntry,
    p: &wrela_abi::manifest::Pipeline,
) -> wrela_gpu::BindingType {
    use wrela_abi::metal::Entry;
    use wrela_gpu::BindingType as T;
    match e.entry {
        Entry::Uniforms { space, size } => match space {
            UniformSpace::Uniform => T::Uniform { dynamic: true, size: u64::from(size) },
            // A uniform block held in storage (Vulkan's two-row matrices) is read at a dynamic
            // offset too: the ring's, as a uniform's.
            UniformSpace::Storage => {
                T::Storage { read_only: true, dynamic: true, size: u64::from(size) }
            }
        },
        Entry::DebugFlag => T::Storage { read_only: false, dynamic: false, size: 0 },
        Entry::Sizes { .. } => T::Uniform { dynamic: false, size: 0 },
        Entry::Binding(kind) => match kind {
            BindingKind::Read | BindingKind::ReadWrite => {
                T::Storage { read_only: kind == BindingKind::Read, dynamic: false, size: 0 }
            }
            BindingKind::Texture
            | BindingKind::DepthTexture
            | BindingKind::Texture3d
            | BindingKind::TextureArray
            | BindingKind::TextureCube
            | BindingKind::DepthTextureArray => {
                let b = p.bindings.iter().find(|b| b.binding == e.binding);
                let sample_type = match b {
                    _ if kind.is_depth() => wrela_gpu::SampleType::Depth,
                    Some(b) => sample_type(colour_format(b)),
                    None => wrela_gpu::SampleType::Float { filterable: true },
                };
                T::Texture { sample_type, dimension: view_dimension(kind) }
            }
            BindingKind::Sampler => T::Sampler { comparison: false, filtering: true },
            BindingKind::ComparisonSampler => T::Sampler { comparison: true, filtering: true },
            BindingKind::NonFilteringSampler => T::Sampler { comparison: false, filtering: false },
            // Only a kernel writes a texture's texels.
            BindingKind::StorageTexture
            | BindingKind::StorageTextureArray
            | BindingKind::StorageTexture3d => {
                let format = p
                    .bindings
                    .iter()
                    .find(|b| b.binding == e.binding)
                    .map_or(TextureFormat::Rgba8, colour_format);
                T::StorageTexture { format, dimension: view_dimension(kind) }
            }
        },
    }
}

impl Builder {
    /// Reports the pipeline built of `p` on stderr, in a line, as it's built
    /// ([`crate::pipeline_report`]).
    fn report(&self, p: &wrela_abi::manifest::Pipeline, source: &[u8], kind: &Kind, ms: f64) {
        use crate::pipeline_report::{Report, line, metal};
        let report = match (&p.stage, kind) {
            (_, Kind::Unavailable) => Report::Unavailable { features: &p.requires },
            (Stage::Compute { entry, workgroup_size }, _) => Report::Compute {
                workgroup: *workgroup_size,
                metal: match (wrela_abi::Target::HERE, std::str::from_utf8(source)) {
                    (wrela_abi::Target::Metal, Ok(text)) => metal(&self.device, text, entry),
                    _ => Err("only Metal reports more".into()),
                },
                device_max: self.limits.max_compute_invocations_per_workgroup,
            },
            (Stage::Render { vertex_entry, fragment_entry, .. }, _) => Report::Render {
                vertex: vertex_entry,
                fragment: fragment_entry,
                variants: match kind {
                    Kind::Render { variants, .. } => variants.len(),
                    _ => 0,
                },
            },
        };
        let bytes = source.len();
        eprintln!("{}  shader {bytes} B  built in {ms:.1} ms", line(&p.shader, &p.name, &report));
    }
}

/// A texture made as `desc` says (2D, one level and one layer: a pass's target), and its view.
fn attachment(
    device: &wrela_gpu::Device,
    desc: &wrela_gpu::TextureDesc<'_>,
) -> (wrela_gpu::Texture, wrela_gpu::TextureView) {
    let texture = device.create_texture(desc);
    let view = texture.create_view(&wrela_gpu::ViewDesc {
        dimension: wrela_gpu::ViewDimension::D2,
        base_level: 0,
        level_count: Some(1),
        base_layer: 0,
        layer_count: Some(1),
    });
    (texture, view)
}

impl Gpu {
    /// Makes the screen `width` x `height` in its format ([`Gpu::screen_format`]), a new, cleared
    /// texture.
    fn make_screen(&mut self, width: u32, height: u32) {
        let format = self.screen_format;
        let (texture, view) = attachment(
            &self.device,
            &wrela_gpu::TextureDesc {
                label: "screen",
                width,
                height,
                depth_or_layers: 1,
                levels: 1,
                dimension: wrela_gpu::TextureDimension::D2,
                format,
                usage: wrela_gpu::TextureUsage {
                    render: true,
                    copy_src: true,
                    ..Default::default()
                },
            },
        );
        self.screen = Some(Screen { texture, view, width, height, format });
    }

    /// The screen's format: what it is, or is made in once there's one.
    fn screen_format(&self) -> TextureFormat {
        self.screen_format
    }

    /// Every render pipeline that draws the screen, built for the screen's format: those of a
    /// build taken after an HDR screen was shown, as well as those it was shown with.
    fn screen_variants(&mut self) -> Result<()> {
        let (device, format) = (self.device.clone(), self.screen_format);
        for p in &mut self.pipelines {
            let depths = p.screen_depths.clone();
            for depth in depths {
                let targets = (vec![format], depth.then_some(TextureFormat::Depth32Float));
                render_variant(&device, &p.name, &p.layout, &mut p.kind, targets).map_err(|e| {
                    Error::Gpu(format!("`{}` for the {format:?} screen: {e}", p.name))
                })?;
            }
        }
        Ok(())
    }

    /// Makes the screen `width` x `height` (a new, cleared texture if the size changed).
    pub(crate) fn set_screen(&mut self, width: u32, height: u32) -> Result<()> {
        let max = self.limits.max_texture_dimension_2d;
        if width == 0 || height == 0 || width > max || height > max {
            return Err(Error::Gpu(format!(
                "a {width}x{height} screen is outside 1..={max} in each dimension"
            )));
        }
        if self.screen.as_ref().is_some_and(|s| (s.width, s.height) == (width, height)) {
            return Ok(());
        }
        self.make_screen(width, height);
        Ok(())
    }

    /// How the screen is shown now: `SCREEN_LIMIT`'s bits.
    pub(crate) fn screen_bits(&self) -> u32 {
        self.program_limits.screen
    }

    /// The screen as linear floats, RGBA rows top to bottom (waits for the GPU).
    pub(crate) fn read_screen_floats(&mut self) -> Result<Vec<f32>> {
        self.check_debug_flag_now()?;
        self.flush()?;
        let screen =
            self.screen.as_ref().ok_or_else(|| Error::Gpu("there's no screen yet".into()))?;
        let (w, h, format) = (screen.width, screen.height, screen.format);
        let raw = read_rows(
            &self.device,
            &mut self.image_readback,
            &screen.texture,
            0,
            [0, 0, 0],
            [w, h],
            format,
        )?;
        let texel = format.bytes_per_texel() as usize;
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for t in raw.chunks_exact(texel) {
            if format == TextureFormat::Rgba16Float {
                out.extend(
                    (0..4).map(|k| f16_to_f32(u16::from_le_bytes([t[2 * k], t[2 * k + 1]]))),
                );
            } else {
                out.extend(t.iter().map(|&b| f32::from(b) / 255.0));
            }
        }
        Ok(out)
    }

    /// Lets an offscreen screen go HDR when the program asks (a test's `hdr=1`): else, with no
    /// window, it stays SDR in sRGB and the program's goldens with it.
    pub(crate) fn allow_offscreen_hdr(&mut self, on: bool) {
        self.offscreen_hdr = on;
    }

    /// Between frames: the screen shown as the program last asked (`ConfigureScreen`), as far as
    /// the window (or, offscreen, a test) allows. HDR, its screen is half floats, every render
    /// pipeline that draws it built for them first; then `screen_is` says what's done.
    fn show_screen(&mut self) -> Result<()> {
        let Some((hdr, p3)) = self.screen_asked.take() else { return Ok(()) };
        let device = self.device.clone();
        let (hdr, p3) = match &mut self.presenter {
            Some(p) => {
                let done = p.surface.set_space(hdr, p3);
                // The present pass draws the surface's frames: in their new format, if it moved.
                if p.surface.format() != p.format {
                    let (pipeline, layout) = present_pipeline(&device, p.surface.format())?;
                    (p.pipeline, p.layout, p.format) = (pipeline, layout, p.surface.format());
                }
                done
            }
            None if self.offscreen_hdr => (hdr, p3),
            None => (false, false),
        };
        let format = if hdr { TextureFormat::Rgba16Float } else { SCREEN_FORMAT };
        if format != self.screen_format {
            // Every pipeline that draws the screen, built for its new format first; a screen not
            // made yet is made in it (`Gpu::set_screen`).
            self.screen_format = format;
            self.screen_variants()?;
            if let Some(screen) = &self.screen {
                let (w, h) = (screen.width, screen.height);
                self.screen = None;
                self.make_screen(w, h);
            }
        }
        self.program_limits.screen = u32::from(hdr) | u32::from(p3) << 1;
        if !hdr {
            self.program_limits.headroom = 1000;
        }
        Ok(())
    }

    /// Shows the screen in `window` from now on, `width` × `height`, presenting as `present`
    /// says: each frame, [`Gpu::present`] draws it into the window's next frame.
    pub(crate) fn attach_window(
        &mut self,
        window: Arc<dyn wrela_gpu::Window>,
        width: u32,
        height: u32,
        present: wrela_gpu::PresentMode,
    ) -> Result<()> {
        let gpu_err = |e: wrela_gpu::Error| Error::Gpu(e.0);
        let mut surface = self.device.create_surface(window.clone()).map_err(gpu_err)?;
        let config = wrela_gpu::SurfaceConfig { width, height, present, frames: 3 };
        surface.configure(&config).map_err(gpu_err)?;
        let (pipeline, layout) = present_pipeline(&self.device, surface.format())?;
        let format = surface.format();
        self.presenter = Some(Presenter {
            window,
            id_base: 0,
            next_id: 0,
            surface,
            format,
            config,
            pipeline,
            layout,
        });
        Ok(())
    }

    /// The window is now `width` × `height`.
    pub(crate) fn resize_window(&mut self, width: u32, height: u32) -> Result<()> {
        let Some(p) = &mut self.presenter else { return Ok(()) };
        if (p.config.width, p.config.height) == (width, height) || width == 0 || height == 0 {
            return Ok(());
        }
        p.config.width = width;
        p.config.height = height;
        p.surface.configure(&p.config).map_err(|e| Error::Gpu(e.0))
    }

    /// The frames the window showed since the last take: each one's id ([`Gpu::present`]) and
    /// when it reached the display.
    pub(crate) fn take_presented(&mut self) -> Vec<(u64, std::time::Instant)> {
        self.presenter
            .as_mut()
            .map(|p| {
                let base = p.id_base;
                p.surface.take_presented().into_iter().map(|(id, at)| (id + base, at)).collect()
            })
            .unwrap_or_default()
    }

    /// Draws the screen into the window's next frame and shows it: the frame's id (`None` when
    /// the window changed and the frame was skipped).
    pub(crate) fn present(&mut self) -> Result<Option<u64>> {
        self.flush()?;
        let device = self.device.clone();
        let (Some(p), Some(screen)) = (&mut self.presenter, &self.screen) else {
            return Ok(None);
        };
        let gpu_err = |e: wrela_gpu::Error| Error::Gpu(e.0);
        let frame = match p.surface.acquire() {
            Ok(f) => f,
            // The window changed under it: configured again, the frame is skipped.
            Err(_) => {
                return p.surface.configure(&p.config).map(|()| None).map_err(gpu_err);
            }
        };
        if self.program_limits.screen & 1 != 0 {
            self.program_limits.headroom = (p.surface.headroom().max(1.0) * 1000.0) as u32;
        }
        let cb = draw_presented(&device, &p.pipeline, &p.layout, &screen.view, &frame.view);
        device.submit_frame(vec![cb], &frame);
        let id = p.surface.present(frame).map_err(gpu_err)? + p.id_base;
        p.next_id = id + 1;
        Ok(Some(id))
    }

    /// How many times the device has been made again (`std::gpu::device_epoch`).
    pub(crate) fn device_epoch(&self) -> u32 {
        self.program_limits.epoch
    }

    /// A frame begins: no label carries over from the last one.
    pub(crate) fn set_frame(&mut self, frame: usize) {
        self.frame = frame;
        self.label = None;
    }

    /// What it did since the last take.
    pub(crate) fn take_counters(&mut self) -> Counters {
        std::mem::take(&mut self.counters)
    }

    pub(crate) fn take_timings(&mut self) -> Result<Vec<GpuTiming>> {
        self.flush()?;
        self.collect()?;
        Ok(self.timer.as_mut().map(|t| std::mem::take(&mut t.results)).unwrap_or_default())
    }

    /// The GPU's name, API, device type, PCI IDs and driver.
    pub(crate) fn adapter(&self) -> String {
        let info = self.device.info();
        format!(
            "{} ({:?}, {:?}, {:04x}:{:04x}; {}, {})",
            info.name,
            info.backend,
            info.device_type,
            info.vendor,
            info.device,
            info.driver,
            info.driver_info
        )
    }

    /// Reports (and clears) errors the GPU reported: a failed command buffer, a validation
    /// layer's message, a lost device.
    fn check_errors(&self) -> Result<()> {
        let errors = self.device.take_errors();
        if errors.is_empty() {
            return Ok(());
        }
        Err(Error::Gpu(errors.join("\n")))
    }

    fn buffer(&self, handle: u32) -> &wrela_gpu::Buffer {
        match &self.resources[&handle] {
            Resource::Buffer(b) => b,
            _ => unreachable!("the checker checked {handle} is a buffer"),
        }
    }

    /// A texture pass's targets' size: its first colour target's level, or its depth's.
    fn pass_size(&self, pass: &Pass) -> (u32, u32) {
        let (handle, level) = match pass.colors().first() {
            Some(c) => (c.texture, c.level),
            None => (pass.depth, 0),
        };
        let t = &self.texture(handle).texture;
        stream::level_size(t.width(), t.height(), level)
    }

    /// The texture `handle`.
    fn texture(&self, handle: u32) -> &Texture {
        match &self.resources[&handle] {
            Resource::Texture(t) => t,
            _ => unreachable!("the checker checked {handle} is a texture"),
        }
    }

    /// Bind group 0 of `pipeline` with these bindings, from the cache or made now.
    fn bind_group(&mut self, pipeline: u32, bindings: &[Binding]) -> wrela_gpu::BindGroup {
        if let Some(bg) = self.bind_groups[pipeline as usize].get(bindings) {
            return bg.clone();
        }
        let p = &self.pipelines[pipeline as usize];
        // Each binding's resource, with a texture's view made first: an entry borrows its view.
        let resources: Vec<(&Resource, Option<wrela_gpu::TextureView>)> = bindings
            .iter()
            .zip(&p.bindings)
            .map(|(b, slot)| {
                let resource = &self.resources[&b.handle];
                let view = match resource {
                    Resource::Texture(t) => {
                        let view = View::of_words(b.offset, b.size);
                        Some(t.view(view, view_dimension(slot.kind), slot.kind.is_storage()))
                    }
                    _ => None,
                };
                (resource, view)
            })
            .collect();
        let mut entries = Vec::with_capacity(
            bindings.len() + usize::from(p.uniform.is_some()) + usize::from(p.debug_flag.is_some()),
        );
        if let Some((binding, size)) = p.uniform {
            entries.push(wrela_gpu::BindGroupEntry {
                binding,
                resource: wrela_gpu::Resource::Buffer {
                    buffer: &self.ring.buffer,
                    offset: 0,
                    size: Some(u64::from(size)),
                },
            });
        }
        for ((b, slot), (resource, view)) in bindings.iter().zip(&p.bindings).zip(&resources) {
            let resource = match (resource, view) {
                (_, Some(view)) => wrela_gpu::Resource::View(view),
                (Resource::Buffer(buffer), None) => wrela_gpu::Resource::Buffer {
                    buffer,
                    offset: u64::from(b.offset),
                    // A size of 0 binds the rest of the buffer.
                    size: (b.size != 0).then_some(u64::from(b.size)),
                },
                (Resource::Sampler(s), None) => wrela_gpu::Resource::Sampler(s),
                (Resource::Texture(_), None) => unreachable!("a texture's view is made above"),
            };
            entries.push(wrela_gpu::BindGroupEntry { binding: slot.binding, resource });
        }
        if let (Some(binding), Some(flag)) = (p.debug_flag, &self.debug_flag) {
            let resource =
                wrela_gpu::Resource::Buffer { buffer: &flag.buffer, offset: 0, size: None };
            entries.push(wrela_gpu::BindGroupEntry { binding, resource });
        }
        // The bound size of each storage buffer the shader reads the length of
        // (`wrela_abi::metal`), in a uniform buffer of this bind group's own, which the bind
        // group keeps alive.
        // The flag's buffer: its one word, or a counting build's tallies after it.
        let flag_bytes = self.debug_flag.as_ref().map_or(4, |f| (4 * f.words) as u32);
        let sizes = p.sizes.as_ref().map(|s| {
            let words = s.buffers.iter().map(|&binding| {
                if Some(binding) == p.debug_flag {
                    return flag_bytes;
                }
                let at = p.bindings.iter().position(|b| b.binding == binding);
                let (b, resource) = match at {
                    Some(k) => (&bindings[k], &resources[k].0),
                    None => return 0,
                };
                match resource {
                    Resource::Buffer(buffer) if b.size == 0 => {
                        u32::try_from(buffer.size().saturating_sub(u64::from(b.offset)))
                            .unwrap_or(u32::MAX)
                    }
                    _ => b.size,
                }
            });
            (s.binding, sizes_buffer(&self.device, "buffer sizes", words))
        });
        if let Some((binding, buffer)) = &sizes {
            entries.push(wrela_gpu::BindGroupEntry {
                binding: *binding,
                resource: wrela_gpu::Resource::Buffer { buffer, offset: 0, size: None },
            });
        }
        let bg = self.device.create_bind_group(&p.layout, &entries);
        // The reverse index holds this key once per resource. Share its binding array rather
        // than copying the whole array for every resource in it.
        let key: Arc<[Binding]> = bindings.into();
        for b in bindings {
            self.bound_in.entry(b.handle).or_default().insert((pipeline, key.clone()));
        }
        self.bind_groups[pipeline as usize].insert(key, bg.clone());
        bg
    }

    /// Forgets a destroyed resource, and the cached bind groups it's in. Work already recorded
    /// holds its own references until the GPU is done.
    fn destroy(&mut self, handle: u32) {
        if self.gone.len() >= GONE_KEPT {
            self.gone.clear();
        }
        self.gone.insert(handle, (self.frame, self.calls));
        self.resources.remove(&handle);
        self.made.remove(&handle);
        for group in self.bound_in.remove(&handle).unwrap_or_default() {
            for other in group.1.iter().filter(|b| b.handle != handle) {
                if let Some(groups) = self.bound_in.get_mut(&other.handle) {
                    groups.remove(&group);
                }
            }
            self.bind_groups[group.0 as usize].remove(&group.1);
        }
    }

    /// Makes room for `bytes` more uniform bytes (counted with alignment) in this submission,
    /// flushing first or growing the ring if it must. Call before recording the work that uses
    /// them, since a flush submits what's recorded.
    fn reserve_uniforms(&mut self, opcode: Opcode, bytes: u64) -> Result<()> {
        let start = align_to(self.ring.staging.len() as u64, self.align);
        if start + bytes <= self.ring.capacity() {
            return Ok(());
        }
        self.flush()?;
        // Grown whenever it runs out, not only for a request larger than it: a ring a frame
        // overflows submits, and waits, again every frame (64 walking bodies, 1,284 draws: the
        // Radeon's frame spent most of its 250 ms of recording waiting so). Doubled, a frame's
        // uniforms fit in a few frames' time and are submitted once.
        if start + bytes > self.ring.capacity() {
            if !self.ring.grow_to_fit(&self.device, bytes, self.limits.max_buffer_size) {
                return Err(Error::command(
                    opcode,
                    format!("{bytes} uniform bytes in one submission is over the limit"),
                ));
            }
            // Cached bind groups point at the old ring.
            self.bind_groups.iter_mut().for_each(HashMap::clear);
            self.bound_in.clear();
        }
        Ok(())
    }

    /// Appends uniform bytes to this submission's slice of the ring; returns their offset.
    /// [`Gpu::reserve_uniforms`] must have made room.
    fn push_uniforms(&mut self, bytes: &[u8]) -> Option<u32> {
        if bytes.is_empty() {
            return None;
        }
        let offset = align_to(self.ring.staging.len() as u64, self.align);
        debug_assert!(offset + bytes.len() as u64 <= self.ring.capacity());
        self.ring.staging.resize(offset as usize, 0);
        self.ring.staging.extend_from_slice(bytes);
        Some(offset as u32)
    }

    fn aligned(&self, len: usize) -> u64 {
        align_to(len as u64, self.align)
    }

    /// The pending command encoder, taken out so a pass can record into it while `self` is
    /// borrowed. Put it back (`self.encoder = Some(..)`) before anything can flush.
    fn take_encoder(&mut self) -> wrela_gpu::CommandEncoder {
        self.encoder.take().unwrap_or_else(|| self.device.create_command_encoder("frame"))
    }

    /// Submits everything recorded so far, after writing its uniform bytes (a pass that ended
    /// and waits is recorded first).
    pub(crate) fn flush(&mut self) -> Result<()> {
        self.submit(true)
    }

    /// [`Gpu::flush`]; with `collect` false, a timed run's timestamps wait for the next flush
    /// that collects (a submission split off by its budget, `Gpu::room`, adds no wait for them to
    /// a frame, so its span is the GPU's as it was in one submission).
    fn submit(&mut self, collect: bool) -> Result<()> {
        self.record_ended()?;
        let queued = std::mem::take(&mut self.queued);
        let Some(encoder) = self.encoder.take() else {
            debug_assert!(self.ring.staging.is_empty() && self.upload.staging.is_empty());
            return Ok(());
        };
        self.ring.write(&self.device, &mut self.counters);
        self.upload.write(&self.device, &mut self.counters);
        let index = self.device.submit(vec![encoder.finish()]);
        self.submissions += 1;
        for s in self.settling.iter_mut().filter(|s| s.done_at.is_none()) {
            s.done_at = Some(index);
        }
        if self.device.has_clock() {
            let (alone, pieces) = match self.job_piece.take() {
                Some(p) => (true, vec![p]),
                None => (false, std::mem::take(&mut self.job_light)),
            };
            let dispatches = queued.bound_ms > 0.0;
            let stamp = if alone { self.job_stamp.take() } else { None };
            let job = Job {
                index,
                frame: self.frame,
                call: self.calls,
                pieces,
                alone,
                dispatches,
                stamp,
            };
            self.jobs.push_back(job);
        }
        self.measure()?;
        self.largest.commands = self.largest.commands.max(queued.commands);
        self.largest.bound_ms = self.largest.bound_ms.max(queued.bound_ms);
        if crate::step::logging() {
            eprintln!(
                "step frame {} submit | dispatches and passes {} bound {:.1} ms",
                self.frame, queued.commands, queued.bound_ms
            );
        }
        // The serial mode reads its timestamps once a frame (`collect`), so nothing waits
        // between its submissions and the GPU keeps its clocks.
        if collect && self.timer.as_ref().is_some_and(|t| !t.serial) {
            self.collect()?;
        }
        Ok(())
    }

    /// Starts capturing the device's work to `path`, a `.gputrace` ([`crate::capture`]): what's
    /// recorded so far is submitted first, so the capture holds only what comes after.
    pub(crate) fn begin_capture(&mut self, path: &std::path::Path) -> Result<()> {
        self.flush()?;
        crate::capture::start(&self.device, path).map_err(Error::Gpu)
    }

    /// Submits what's recorded, waits for the GPU to finish it, and stops the capture, which
    /// writes its file.
    pub(crate) fn end_capture(&mut self) -> Result<()> {
        let done = self.flush().map(|()| wait(&self.device));
        crate::capture::stop(&self.device);
        done
    }

    /// Reads back the timestamps of the submissions made since the last call.
    fn collect(&mut self) -> Result<()> {
        if let Some(timer) = &mut self.timer
            && !timer.pending.is_empty()
        {
            timer.collect(&self.device, &mut self.asked)?;
        }
        Ok(())
    }

    /// Reads the time of every submission done and not yet measured, and holds each to the line.
    fn measure(&mut self) -> Result<()> {
        let done = self.device.completed();
        while self.jobs.front().is_some_and(|j| j.index <= done) {
            let job = self.jobs.pop_front().expect("checked above");
            let stamp = job.stamp;
            let judged = match self.device.gpu_ms(job.index) {
                Some(ms) => self.judge(job, ms),
                None => Ok(()),
            };
            // Its stamps are read: the slot may take another piece's.
            if let Some(slot) = stamp {
                self.stamp_held[slot as usize] = false;
            }
            judged?;
        }
        Ok(())
    }

    /// Learns a piece's cost from its time `ms`, and fails a job over the line: a piece of a
    /// pass alone over it (the host cut the pass by what its draws took before, and they took
    /// far more), or a submission of other work over `SUBMIT_LINES` lines (what `Gpu::room`
    /// holds dispatches to). The light passes of a submission over a quarter of the line are each
    /// measured alone at their next drawing.
    fn judge(&mut self, job: Job, ms: f64) -> Result<()> {
        #[cfg(test)]
        self.measured.push((job.alone.then(|| job.pieces[0].clone()), ms));
        let line = self.line_ms;
        if !job.alone {
            // Light passes that took long together: each is measured alone at its next drawing
            // (at once where the submission held nothing else, else at most once in
            // `AGAIN_AFTER` frames: dispatches may be what took long); over the line, each is
            // costed as if it took the whole of it, so none shares a submission again until it
            // is measured alone.
            if ms > LIGHT_JOB * line {
                for p in &job.pieces {
                    let c = self.costs.entry(p.label.clone()).or_default();
                    c.at(job.call);
                    if !job.dispatches || job.call >= c.alone + AGAIN_AFTER {
                        c.again = true;
                    }
                    if ms > line {
                        c.worst = c.worst.max(ms / f64::from(p.of.max(1)));
                    }
                }
            }
            // A submission of dispatches is held to `SUBMIT_LINES` lines by their bounds: past
            // them, a bound didn't hold. (Passes that grew past what they took are cut from next
            // time.)
            if ms > SUBMIT_LINES * line && job.pieces.is_empty() {
                return Err(Error::Gpu(format!(
                    "a submission of frame {} held the GPU {ms:.1} ms, over twice the line of \
                     {line:.0} ms (AGENTS.md, rule 2), with no passes. The run stops here: a \
                     dispatch's bound didn't hold it",
                    job.frame
                )));
            }
            return Ok(());
        }
        // Over the line by its submission's span: judged by its pass's own begin and end where it
        // was stamped (on Metal the span starts near the commit, so a wait before the pass ran,
        // the GPU waking after a long stretch of CPU work, is in the span and not in the pass).
        let ms = match (ms > line, job.stamp, &self.stamps) {
            (true, Some(slot), Some(stamps)) => match stamped_ms(&self.device, stamps, slot) {
                Some(own) => {
                    if own <= line {
                        eprintln!(
                            "gpu: the pass `{}` took {own:.1} ms of its submission's {ms:.1} on the \
                             GPU's clock (frame {}): judged by its own",
                            job.pieces[0].label, job.frame
                        );
                    }
                    own.min(ms)
                }
                None => ms,
            },
            _ => ms,
        };
        let p = &job.pieces[0];
        let n = p.draws.len().max(1) as f64;
        let idle = self.device.idle_before_ms(job.index);
        let c = self.costs.entry(p.label.clone()).or_default();
        c.at(job.call);
        if ms <= line {
            c.usual = if c.usual > 0.0 { c.usual.min(ms / n) } else { ms / n };
        }
        // Far over what it usually takes: said, with how long the GPU had been idle before it
        // (a GPU left idle lowers its clock, and its next pass runs slow for real).
        let usual = if c.usual > 0.0 && ms / n >= USUAL_TIMES * c.usual {
            format!(
                ". It ran {:.0} times its usual {:.2} ms a draw{}",
                ms / n / c.usual,
                c.usual,
                idle.map_or(String::new(), |i| format!(
                    ": the GPU had been idle {i:.0} ms before it (an idle GPU lowers its clock; \
                     --keep-clocked keeps it up)"
                ))
            )
        } else {
            String::new()
        };
        c.now = c.now.max(ms / n);
        c.again = false;
        c.alone = job.call;
        // A piece alone over the line, but not by much, is measured alone again before the run
        // ends: a submission's span on the M4's clock starts near its commit, so a wait before
        // the pass ran (the GPU waking after a long stretch of CPU work, another app's work, the
        // compositor) is charged to it, a fixed 50-60 ms whatever the pass draws (the engine's
        // judge, 2026-11-20: a draw scissored to 32 px "held the GPU 59 ms"). Twice running,
        // or once over `RECHECK_LINES` lines, it's the pass's own.
        if ms > line && ms <= RECHECK_LINES * line && !c.suspect {
            c.suspect = true;
            eprintln!(
                "gpu: the pass `{}` took {ms:.1} ms in frame {} on the GPU's clock, over the line \
                 of {line:.0} ms: measured alone again before the run ends (a wait in its span is \
                 charged to it){usual}",
                p.label, job.frame
            );
            return Ok(());
        }
        if ms <= line {
            c.suspect = false;
        }
        if ms > line {
            let which = if p.draws.len() == 1 {
                format!("draw {} of its {}", p.draws.start, p.of)
            } else {
                format!("draws {}..{} of its {}", p.draws.start, p.draws.end, p.of)
            };
            let cut = if p.draws.len() == 1 {
                "One draw can't be cut by the host: draw less in it (fewer pixels, or less work \
                 a pixel), or draw it over frames"
            } else {
                "The host cut the pass by what its draws took before; these took far more"
            };
            return Err(Error::Gpu(format!(
                "the pass `{}` held the GPU {ms:.1} ms in frame {} ({which}), over the line of \
                 {line:.0} ms (AGENTS.md, rule 2){usual}. The run stops here, so it can't again. \
                 {cut}",
                p.label, job.frame
            )));
        }
        Ok(())
    }

    /// Makes room for one timed pass's queries in this submission, flushing first if they're
    /// used up. Call before recording the pass (a flush submits what's recorded).
    fn reserve_timestamps(&mut self) -> Result<()> {
        if self.timer.as_ref().is_some_and(|t| t.pending.len() as u32 * 2 >= t.capacity) {
            self.flush()?;
            self.collect()?;
            // Grown whenever it runs out, as the uniform ring is: a frame that used every query
            // waited for the GPU in its middle, the GPU then idle while the CPU recorded the rest
            // (SCENE 10's 269 timed passes a frame: 1.7 to 2.8 ms of idle after the sun's
            // certificates, inside its span, where 309 had put the wait where little followed).
            let device = &self.device;
            if let Some(t) = &mut self.timer {
                t.grow(device);
            }
        }
        Ok(())
    }

    fn create_buffer(&mut self, handle: u32, size: u32) {
        self.made.insert(handle, Made::Buffer { size });
        let buffer = self.device.create_buffer(&wrela_gpu::BufferDesc {
            label: &format!("buffer {handle}"),
            size: u64::from(size),
            mappable: false,
        });
        self.resources.insert(handle, Resource::Buffer(buffer));
    }

    /// A texture `size[0]` × `size[1]` of `kind`, `size[2]` deep (a 3D one) or layers (an
    /// array's, a cube's 6, a 2D one's 1), with `levels` mip levels.
    fn create_texture(
        &mut self,
        handle: u32,
        size: [u32; 3],
        format: TextureFormat,
        writable: bool,
        kind: TextureKind,
        levels: u32,
    ) {
        self.made.insert(handle, Made::Texture { size, format, writable, kind, levels });
        let [width, height, depth] = size;
        let three = kind == TextureKind::D3;
        let features = self.device.features();
        let usage = wrela_gpu::TextureUsage {
            sampled: true,
            copy_src: true,
            render: !three && format.info().renderable.on(|f| features.has(f)),
            copy_dst: !format.is_depth(),
            // Only where kernels write it: a storage texture may give up the GPU's compression
            // of what's drawn into it.
            storage: writable,
        };
        let texture = self.device.create_texture(&wrela_gpu::TextureDesc {
            label: &format!("texture {handle}"),
            width,
            height,
            depth_or_layers: depth.max(1),
            levels,
            dimension: if three {
                wrela_gpu::TextureDimension::D3
            } else {
                wrela_gpu::TextureDimension::D2
            },
            format,
            usage,
        });
        let views = std::cell::RefCell::new(HashMap::new());
        self.resources.insert(handle, Resource::Texture(Texture { texture, format, views }));
    }

    fn create_sampler(
        &mut self,
        handle: u32,
        linear: bool,
        repeat: bool,
        compare: Option<Compare>,
        mip_linear: bool,
        anisotropy: u32,
    ) {
        self.made.insert(handle, Made::Sampler { linear, repeat, compare, mip_linear, anisotropy });
        let sampler = self.device.create_sampler(&wrela_gpu::SamplerDesc {
            linear,
            repeat,
            mip_linear,
            anisotropy: anisotropy.clamp(1, 16) as u16,
            compare,
        });
        self.resources.insert(handle, Resource::Sampler(sampler));
    }

    /// Records a copy from the upload ring ([`Ring`]): the stream's writes are whole words, as a
    /// copy moves. A very big one goes through the queue after what's recorded is submitted.
    fn write_buffer(&mut self, handle: u32, offset: u32, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let len = data.len() as u64;
        if len > u64::from(stream::UPLOAD_MAX) {
            // Work recorded before this write must see the old contents.
            self.flush()?;
            self.device.write_buffer(self.buffer(handle), u64::from(offset), data);
            self.counters.write_buffers += 1;
            self.counters.write_buffer_bytes += len;
            return Ok(());
        }
        let at = self.reserve_upload(len, 1)?;
        self.upload.staging.extend_from_slice(data);
        let mut encoder = self.take_encoder();
        encoder.copy_buffer_to_buffer(
            &self.upload.buffer,
            at,
            self.buffer(handle),
            u64::from(offset),
            len,
        );
        self.encoder = Some(encoder);
        Ok(())
    }

    /// Makes room for an upload, flushing first if necessary. Texture copies need both their
    /// rows and their starting offsets aligned; buffer copies already carry aligned lengths.
    fn reserve_upload(&mut self, bytes: u64, alignment: u64) -> Result<u64> {
        let at = align_to(self.upload.staging.len() as u64, alignment);
        if at + bytes <= self.upload.capacity() {
            return Ok(at);
        }
        self.flush()?;
        if bytes > self.upload.capacity() {
            self.upload.grow_to_fit(&self.device, bytes, u64::MAX);
        }
        Ok(0)
    }

    /// Texels into level `level`, layer `layer` of a texture: `size` texels from `origin`, rows
    /// of blocks of a compressed format.
    fn write_texture(
        &mut self,
        handle: u32,
        level: u32,
        layer: u32,
        origin: [u32; 2],
        size: [u32; 2],
        data: &[u8],
    ) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let format = self.texture(handle).format;
        let row = format.bytes_of(size[0], 1);
        let rows = size[1].div_ceil(format.block());
        let padded = align_to(row, u64::from(wrela_gpu::COPY_BYTES_PER_ROW_ALIGNMENT));
        let bytes = padded * u64::from(rows);
        let extent = whole_blocks(format, size[0], size[1]);
        let [x, y] = origin;
        if bytes > u64::from(stream::UPLOAD_MAX) {
            self.flush()?;
            self.device.write_texture(
                self.image_copy(handle, level, layer, x, y),
                data,
                row as u32,
                rows,
                extent,
            );
            self.counters.write_textures += 1;
            return Ok(());
        }
        let at = self.reserve_upload(bytes, u64::from(wrela_gpu::COPY_BYTES_PER_ROW_ALIGNMENT))?;
        self.upload.staging.resize(at as usize, 0);
        if row == padded {
            self.upload.staging.extend_from_slice(data);
        } else {
            self.upload.staging.resize((at + bytes) as usize, 0);
            for (source, target) in data
                .chunks_exact(row as usize)
                .zip(self.upload.staging[at as usize..].chunks_exact_mut(padded as usize))
            {
                target[..row as usize].copy_from_slice(source);
            }
        }
        let mut encoder = self.take_encoder();
        encoder.copy_buffer_to_texture(
            wrela_gpu::BufferCopy {
                buffer: &self.upload.buffer,
                offset: at,
                bytes_per_row: padded as u32,
                rows_per_image: rows,
            },
            self.image_copy(handle, level, layer, x, y),
            extent,
        );
        self.encoder = Some(encoder);
        Ok(())
    }

    /// One level and layer of a texture, as a copy names it, from its corner.
    fn image_copy(
        &self,
        texture: u32,
        level: u32,
        layer: u32,
        x: u32,
        y: u32,
    ) -> wrela_gpu::TextureCopy<'_> {
        wrela_gpu::TextureCopy {
            texture: &self.texture(texture).texture,
            level,
            origin: [x, y, layer],
            depth_only: false,
        }
    }

    fn copy_texture(&mut self, source: &TexelRect, destination: &stream::TexelAt) {
        let mut encoder = self.take_encoder();
        let extent = whole_blocks(self.texture(source.texture).format, source.width, source.height);
        let (s, d) = (source, destination);
        encoder.copy_texture_to_texture(
            self.image_copy(s.texture, s.level, s.layer, s.x, s.y),
            self.image_copy(d.texture, d.level, d.layer, d.x, d.y),
            extent,
        );
        self.encoder = Some(encoder);
    }

    /// A copy between a texture's rectangle `r` and a buffer's rows from `offset`,
    /// `bytes_per_row` apart: into the buffer (`to_buffer`), or out of it.
    fn copy_rows(
        &mut self,
        r: &TexelRect,
        buffer: u32,
        offset: u32,
        bytes_per_row: u32,
        to_buffer: bool,
    ) {
        let mut encoder = self.take_encoder();
        let format = self.texture(r.texture).format;
        let extent = whole_blocks(format, r.width, r.height);
        let rows = r.height.div_ceil(format.block());
        let rows = wrela_gpu::BufferCopy {
            buffer: self.buffer(buffer),
            offset: u64::from(offset),
            bytes_per_row,
            rows_per_image: rows,
        };
        let image = self.image_copy(r.texture, r.level, r.layer, r.x, r.y);
        if to_buffer {
            encoder.copy_texture_to_buffer(image, rows, extent);
        } else {
            encoder.copy_buffer_to_texture(rows, image, extent);
        }
        self.encoder = Some(encoder);
    }

    fn copy_buffer(&mut self, from: u32, from_offset: u32, to: u32, to_offset: u32, size: u32) {
        let mut encoder = self.take_encoder();
        encoder.copy_buffer_to_buffer(
            self.buffer(from),
            u64::from(from_offset),
            self.buffer(to),
            u64::from(to_offset),
            u64::from(size),
        );
        self.encoder = Some(encoder);
    }

    fn dispatch(
        &mut self,
        pipeline: u32,
        groups: Result<[u32; 3], (u32, u32, u32)>,
        bindings: &[Binding],
        uniforms: &[u8],
    ) -> Result<()> {
        self.label_fresh = false;
        // Weighed before anything is recorded: a dispatch that could hold the GPU too long never
        // reaches it (`crate::step`).
        self.weigh(pipeline, groups)?;
        // Everything that can flush happens before anything is recorded for this dispatch, and
        // the submission's room last: a flush resets what the submission has booked, so a room
        // taken before a reserve that flushes would leave this dispatch uncounted in the next.
        let ms = self.bound_ms(pipeline, groups);
        self.reserve_uniforms(Opcode::Dispatch, self.aligned(uniforms.len()))?;
        self.reserve_timestamps()?;
        self.room(ms)?;
        let offset = self.push_uniforms(uniforms);
        let bind_group = self.bind_group(pipeline, bindings);
        let mut encoder = self.take_encoder();
        // An indirect dispatch's groups are checked on the GPU first, in its own pass
        // (`step::Indirect`): it reads the checked ones, none where they're past what its
        // kernel's bound allows.
        let groups = match groups {
            Ok(g) => Ok(g),
            Err((buffer, at, stated)) => {
                let p = &self.pipelines[pipeline as usize];
                let crate::step::Bound::Most(most) = p.bound else {
                    unreachable!("`weigh` refused a kernel without a bound");
                };
                // The groups the GPU lets through: what the limit allows, and no more than the
                // program said it asks (`most:`), which is what it was weighed and booked as.
                let most = crate::step::most_groups(most, p.per_group, wrela_abi::Target::HERE);
                let most = if stated > 0 { most.min(u64::from(stated)) } else { most };
                let name = p.name.clone();
                let source = self.buffer(buffer).clone();
                let device = self.device.clone();
                match self.indirect.prepare(&device, &name, (buffer, &source, at), most) {
                    Ok(checked) => Err(checked),
                    Err(e) => {
                        self.encoder = Some(encoder);
                        return Err(e);
                    }
                }
            }
        };
        if let Some(f) = &mut self.debug_flag {
            let p = &self.pipelines[pipeline as usize];
            let l = self.label.clone().unwrap_or_else(|| p.name.clone());
            f.labels.insert(pipeline, l);
        }
        let p = &self.pipelines[pipeline as usize];
        let label = self.label.as_deref().unwrap_or(&p.name);
        // In a timed span, a dispatch of the label whose timed run is open joins it (a backend
        // that can: Metal's), timed with it; in a serial run each is alone.
        let joins = self.timer.as_ref().is_some_and(|t| !t.serial && t.keep)
            && self.timed_run.as_deref() == Some(label)
            && encoder.timed_compute_open();
        let timestamps = if joins {
            None
        } else {
            self.timer.as_mut().and_then(|t| t.next_if(&self.asked, self.frame, self.calls, label))
        };
        if timestamps.is_some() {
            self.timed_run = Some(Arc::from(label));
        }
        let Kind::Compute(compute) = &p.kind else {
            unreachable!("the checker checked the kind");
        };
        // The indirect check's own dispatch, before the kernel's (a barrier between them).
        if let Err(checked) = &groups {
            let mut check = encoder.begin_compute_pass("indirect check", None);
            self.indirect.record(&mut check, checked);
        }
        let mut pass = encoder.begin_compute_pass(
            &p.name,
            timestamps.zip(self.timer.as_ref()).map(|(i, t)| wrela_gpu::PassTimestamps {
                queries: &t.queries,
                begin: i,
                end: i + 1,
            }),
        );
        pass.set_pipeline(compute);
        pass.set_bind_group(&bind_group, offset.as_slice());
        self.counters.compute_passes += 1;
        self.counters.dispatches += 1;
        self.counters.pipelines += 1;
        self.counters.bind_groups += 1;
        match groups {
            Ok([x, y, z]) => pass.dispatch(x, y, z),
            Err(checked) => pass.dispatch_indirect(&checked.args, checked.at),
        }
        drop(pass);
        self.encoder = Some(encoder);
        self.alone()
    }

    /// Whether a dispatch of `pipeline` over `groups` (or an indirect dispatch's, read on the
    /// GPU) may run: its kernel's bound times its invocations within the limit (`crate::step`).
    fn weigh(&self, pipeline: u32, groups: Result<[u32; 3], (u32, u32, u32)>) -> Result<()> {
        let p = &self.pipelines[pipeline as usize];
        let invocations = invocations(p, groups);
        if crate::step::logging() {
            let label = self.label.as_deref().unwrap_or(&p.name);
            let n = invocations.map_or_else(|| "indirect".to_string(), |n| n.to_string());
            eprintln!(
                "step frame {} dispatch {label} | {} | invocations {n} bound {}",
                self.frame,
                p.name,
                p.bound.shown()
            );
        }
        crate::step::check(&p.name, &p.bound, invocations, wrela_abi::Target::HERE)
    }

    /// The guard's bound of a dispatch of `pipeline` over `groups`, in ms (`crate::step`): an
    /// indirect dispatch's, the limit, which the GPU holds it to.
    fn bound_ms(&self, pipeline: u32, groups: Result<[u32; 3], (u32, u32, u32)>) -> f64 {
        let p = &self.pipelines[pipeline as usize];
        // An indirect dispatch whose most groups are stated is booked as that many, so a few
        // small ones share a submission as direct ones do (backend.md 72: booked at the limit,
        // every second one closed a submission).
        match (&p.bound, invocations(p, groups)) {
            (crate::step::Bound::Most(most), Some(n)) => {
                crate::step::ms_of(*most, n, wrela_abi::Target::HERE)
            }
            _ => crate::step::limit_ms(),
        }
    }

    /// Makes room in the submission being recorded for one more dispatch (its guard bound `ms`)
    /// or pass (0): what's recorded is submitted first where it would pass `SUBMIT_LINES` or
    /// `SUBMIT_COMMANDS`.
    fn room(&mut self, ms: f64) -> Result<()> {
        if self.queued.full(ms, SUBMIT_LINES * self.line_ms) {
            self.submit(false)?;
        }
        self.queued.commands += 1;
        self.queued.bound_ms += ms;
        Ok(())
    }

    /// How many submissions have been made, and the most any has held.
    #[cfg(test)]
    pub(crate) fn submissions(&self) -> (u64, Queued) {
        (self.submissions, self.largest)
    }

    /// In the serial mode, submits what's recorded and waits for the GPU to finish it, so the
    /// next pass or dispatch starts alone. (Submissions alone aren't enough: on Apple GPUs a
    /// render pass overlaps the kernel submitted before it.) Its timestamps are read once a
    /// frame, so only the wait comes between the submissions. With `--steps`, the wait's wall
    /// time is written after its step, so a run cut short by its timeout names the dispatch
    /// that was slow before the frame's timestamps are read.
    fn alone(&mut self) -> Result<()> {
        if self.timer.as_ref().is_some_and(|t| t.serial) {
            self.flush()?;
            let start = std::time::Instant::now();
            wait(&self.device);
            if crate::step::logging() {
                eprintln!("step waited {:.3} ms", start.elapsed().as_secs_f64() * 1e3);
            }
        }
        Ok(())
    }

    /// Opens `pass`, named by the label in force.
    fn begin_pass(&mut self, pass: Pass) {
        self.label_fresh = false;
        let label = self.label.clone();
        self.pass = Some(OpenPass { pass, draws: Vec::new(), label, uses: None, rect_set: false });
    }

    fn draw(&mut self, pipeline: u32, how: DrawHow, bindings: &[Binding], uniforms: &[u8]) {
        if let Some(f) = &mut self.debug_flag {
            let p = &self.pipelines[pipeline as usize];
            let l = self.pass.as_ref().and_then(|o| o.label.clone());
            f.labels.insert(pipeline, l.unwrap_or_else(|| p.name.clone()));
        }
        if crate::step::logging() {
            let p = &self.pipelines[pipeline as usize];
            let label = self.pass.as_ref().and_then(|o| o.label.as_deref()).unwrap_or(&p.name);
            let n = match how {
                DrawHow::Direct { vertices, instances } => format!("{vertices} x {instances}"),
                _ => "indirect".to_string(),
            };
            eprintln!("step frame {} draw {label} | {} | vertices {n}", self.frame, p.name);
        }
        let draw =
            PendingDraw { pipeline, how, bindings: bindings.to_vec(), uniforms: uniforms.to_vec() };
        self.pass_item(PassItem::Draw(draw));
    }

    fn pass_item(&mut self, item: PassItem) {
        let open = self.pass.as_mut().expect("the sequencer checked a pass is open");
        open.rect_set |= matches!(item, PassItem::Scissor(_) | PassItem::Viewport(_));
        open.draws.push(item);
    }

    /// A pass joins one of the same label, so its time and target are that label's. Joining
    /// keeps WebGPU's resource usage scope too. Two individually valid passes may disagree on
    /// whether a buffer is read-only or read-write, including indirect arguments and indices,
    /// which are read-only. Decide after both passes' draws are known.
    fn can_join(&self, ended: &mut OpenPass, next: &OpenPass) -> bool {
        if !next.pass.joins(&ended.pass) || next.label != ended.label {
            return false;
        }
        if ended.uses.is_none() {
            let mut uses = HashMap::new();
            if !self.add_uses(&ended.draws, &mut uses) {
                return false;
            }
            ended.uses = Some(uses);
        }
        self.add_uses(&next.draws, ended.uses.as_mut().expect("built above"))
    }

    fn add_uses(&self, draws: &[PassItem], uses: &mut HashMap<u32, bool>) -> bool {
        let mut compatible = |handle, write| match uses.insert(handle, write) {
            Some(before) => before == write,
            None => true,
        };
        for item in draws {
            let PassItem::Draw(draw) = item else { continue };
            for (binding, slot) in
                draw.bindings.iter().zip(&self.pipelines[draw.pipeline as usize].bindings)
            {
                if !matches!(
                    slot.kind,
                    BindingKind::Sampler
                        | BindingKind::ComparisonSampler
                        | BindingKind::NonFilteringSampler
                ) && !compatible(
                    binding.handle,
                    slot.kind == BindingKind::ReadWrite || slot.kind.is_storage(),
                ) {
                    return false;
                }
            }
            match draw.how {
                DrawHow::Direct { .. } => {}
                DrawHow::Indirect { buffer, .. } => {
                    if !compatible(buffer, false) {
                        return false;
                    }
                }
                DrawHow::IndexedIndirect { indices, buffer, .. } => {
                    if !compatible(indices, false) || !compatible(buffer, false) {
                        return false;
                    }
                }
                DrawHow::Indexed { indices, .. } => {
                    if !compatible(indices, false) {
                        return false;
                    }
                }
                DrawHow::Multi { indices, buffer, .. } => {
                    if !compatible(buffer, false)
                        || indices.is_some_and(|i| !compatible(i[0], false))
                    {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Ends the open pass: on the screen (`Present`) it's recorded; into textures (`EndPass`),
    /// it waits for the next command, which may continue it (`Pass::joins`, the same label),
    /// unless the timing is serial.
    fn end_pass(&mut self, op: Opcode) -> Result<()> {
        let Some(mut open) = self.pass.take() else {
            unreachable!("the sequencer checked a pass is open");
        };
        let serial = self.timer.as_ref().is_some_and(|t| t.serial);
        if op == Opcode::EndPass && !serial {
            if let Some(mut ended) = self.ended.take() {
                if self.can_join(&mut ended, &open) {
                    // A pass begins with the whole target its scissor and viewport: where the
                    // one it joins set either, the joined draws start from the whole target.
                    if ended.rect_set {
                        let (w, h) = self.pass_size(&ended.pass);
                        ended.draws.push(PassItem::Scissor([0, 0, w, h]));
                        ended
                            .draws
                            .push(PassItem::Viewport([0.0, 0.0, w as f32, h as f32, 0.0, 1.0]));
                    }
                    ended.draws.append(&mut open.draws);
                    ended.rect_set |= open.rect_set;
                    self.ended = Some(ended);
                    return Ok(());
                }
                self.record_pass(Opcode::EndPass, ended)?;
            }
            self.ended = Some(open);
            return Ok(());
        }
        self.record_pass(op, open)
    }

    /// Records the pass that ended and waits (`ended`), if there is one.
    fn record_ended(&mut self) -> Result<()> {
        match self.ended.take() {
            Some(open) => self.record_pass(Opcode::EndPass, open),
            None => Ok(()),
        }
    }

    /// Records a pass's draws: one render pass in a submission of its own, or, where what its
    /// draws took before says it would hold the GPU past a quarter of the line, several, each
    /// a piece of its draws on the target as the one before left it, in its own submission. A
    /// pass whose draws weren't measured yet draws each alone. So no job the GPU runs holds
    /// many heavy draws (2026-11-19: a pass of 18 bodies' edge rays, over the whole screen, held
    /// the Radeon 65 ms in one job; render passes were counted, not weighed).
    fn record_pass(&mut self, op: Opcode, open: OpenPass) -> Result<()> {
        let pass = open.pass;
        let on_screen = pass.on_screen();
        if on_screen && self.screen.is_none() {
            return Err(Error::command(
                op,
                "there's no screen: the host draws only while running frames",
            ));
        }
        let colors: Vec<TextureFormat> = pass
            .colors()
            .iter()
            .map(|c| {
                if c.texture == stream::SCREEN {
                    self.screen_format()
                } else {
                    self.texture(c.texture).format
                }
            })
            .collect();
        let depth_format = (pass.depth != stream::NONE).then(|| self.texture(pass.depth).format);
        // Each draw's pipeline has a variant for these targets, made at load: the checker
        // checked they're among its manifest's.
        let formats = (colors, depth_format);
        let name = open.label.clone().unwrap_or_else(|| {
            let (screen, texture) = &*UNNAMED_PASSES;
            if on_screen { screen } else { texture }.clone()
        });
        let target = match (pass.colors().first(), pass.depth) {
            (Some(c), _) if c.texture == stream::SCREEN => None,
            (Some(c), _) => Some((c.texture, c.level, c.layer)),
            (None, stream::NONE) => None,
            (None, depth) => Some((depth, 0, pass.depth_layer)),
        };
        if let (Some(name), Some(target)) = (&open.label, target) {
            self.labelled.insert(name.clone(), target);
            self.labelled_order.retain(|l| l != name);
            self.labelled_order.push(name.clone());
        }
        let draws = open.draws.iter().filter(|d| matches!(d, PassItem::Draw(_))).count() as u32;
        let cost = self.costs.get(&name).copied();
        let light = cost.filter(|c| !c.again).map(|c| c.per_draw() * f64::from(draws));
        let light = light.filter(|&ms| ms <= LIGHT * self.line_ms);
        let forced = self
            .forced
            .as_ref()
            .filter(|(l, _)| l.as_deref().is_none_or(|l| l == &*name))
            .map(|(_, k)| (*k).min(draws))
            .filter(|&k| k >= 2);
        if forced.is_none() && (draws == 0 || !self.device.has_clock() || light.is_some()) {
            let ms = light.unwrap_or(0.0);
            // Its submission holds light passes that took a quarter of the line at most.
            if self.queued.passes_ms + ms > LIGHT_JOB * self.line_ms {
                self.submit(false)?;
            }
            self.record_piece(op, &pass, &formats, &name, open.draws, true, false)?;
            self.queued.passes_ms += ms;
            if light.is_some() {
                self.job_light.push(Piece { label: name, draws: 0..draws, of: draws });
            }
            self.counters.render_passes += 1;
            return if on_screen { self.flush() } else { self.alone() };
        }
        let ranges = match forced {
            Some(k) => (0..k).map(|j| j * draws / k..(j + 1) * draws / k).collect(),
            None => pieces(draws, cost, self.line_ms / 4.0),
        };
        // What's recorded before it is a submission of its own.
        self.submit(false)?;
        // Each piece's items: the scissor and viewport in force where it starts, then its own.
        let mut parts: Vec<Vec<PassItem>> = vec![Vec::new()];
        let (mut scissor, mut viewport) = (None, None);
        let mut drawn = 0;
        let mut at = 0;
        for item in open.draws {
            match &item {
                PassItem::Scissor(r) => scissor = Some(*r),
                PassItem::Viewport(v) => viewport = Some(*v),
                PassItem::Draw(_) => {
                    if drawn == ranges[at].end {
                        at += 1;
                        let mut next = Vec::new();
                        next.extend(scissor.map(PassItem::Scissor));
                        next.extend(viewport.map(PassItem::Viewport));
                        parts.push(next);
                    }
                    drawn += 1;
                }
            }
            parts.last_mut().expect("one at least").push(item);
        }
        let last = parts.len() - 1;
        for (k, (items, range)) in parts.into_iter().zip(ranges).enumerate() {
            self.record_piece(op, &pass, &formats, &name, items, k == 0, true)?;
            self.job_piece = Some(Piece { label: name.clone(), draws: range, of: draws });
            self.submit(on_screen && k == last)?;
            if k != last {
                self.alone()?;
            }
        }
        self.counters.render_passes += 1;
        if on_screen { Ok(()) } else { self.alone() }
    }

    /// Records `items` as one render pass of `pass` into the pending encoder: the first piece
    /// of a pass loads its targets as the pass says, every later one keeps what's there.
    #[allow(clippy::too_many_arguments)]
    fn record_piece(
        &mut self,
        op: Opcode,
        pass: &Pass,
        formats: &Targets,
        label: &str,
        items: Vec<PassItem>,
        first: bool,
        alone: bool,
    ) -> Result<()> {
        // Everything that can flush happens before anything is recorded for this pass.
        let total = items
            .iter()
            .map(|d| match d {
                PassItem::Draw(d) => self.aligned(d.uniforms.len()),
                _ => 0,
            })
            .sum();
        self.reserve_uniforms(op, total)?;
        self.reserve_timestamps()?;
        // The submission's room last, as a dispatch's (`Gpu::dispatch`): a flush resets what it
        // has booked.
        self.room(0.0)?;
        let timestamps =
            self.timer.as_mut().and_then(|t| t.next_if(&self.asked, self.frame, self.calls, label));
        // A piece measured alone, where no timer stamps it, is stamped for the guard: its pass's
        // own begin and end, without what the GPU did or waited for before it ran.
        // A slot still held by a piece not yet measured (more than `STAMP_RING` pieces in
        // flight) isn't given: this piece goes unstamped, judged by its span.
        let free = !self.stamp_held[self.stamp_next as usize];
        let guard_stamp =
            (alone && timestamps.is_none() && self.stamps.is_some() && free).then(|| {
                let slot = self.stamp_next;
                self.stamp_held[slot as usize] = true;
                self.stamp_next = (self.stamp_next + 1) % STAMP_RING;
                slot
            });
        self.job_stamp = guard_stamp;
        let mut prepared = Vec::with_capacity(items.len());
        for d in &items {
            if let PassItem::Draw(d) = d {
                let offset = self.push_uniforms(&d.uniforms);
                let bind_group = self.bind_group(d.pipeline, &d.bindings);
                prepared.push((bind_group, offset));
            }
        }
        let mut encoder = self.take_encoder();
        let color_views: Vec<wrela_gpu::TextureView> = pass
            .colors()
            .iter()
            .map(|c| match c.texture {
                stream::SCREEN => self.screen.as_ref().expect("checked above").view.clone(),
                h => self.texture(h).target(c.level, c.layer),
            })
            .collect();
        let depth_view = (pass.depth != stream::NONE)
            .then(|| self.texture(pass.depth).target(0, pass.depth_layer));
        let color_attachments: Vec<wrela_gpu::ColorAttachment<'_>> = pass
            .colors()
            .iter()
            .zip(&color_views)
            .map(|(c, view)| {
                let load = if c.keep || !first {
                    wrela_gpu::LoadOp::Load
                } else {
                    wrela_gpu::LoadOp::Clear(c.clear.map(f64::from))
                };
                wrela_gpu::ColorAttachment { view, load }
            })
            .collect();
        let depth_attachment = depth_view.as_ref().map(|view| {
            let load = if pass.keep_depth || !first {
                wrela_gpu::LoadOp::Load
            } else {
                wrela_gpu::LoadOp::Clear(pass.clear_depth)
            };
            wrela_gpu::DepthAttachment { view, load }
        });
        // Counted here, and added once the pass, which borrows the executor, has ended.
        let mut counters = Counters::default();
        // The GPU's name for it says its frame, so a failed command buffer names the frame.
        let named = format!("{label} (frame {})", self.frame);
        let mut rp = encoder.begin_render_pass(&wrela_gpu::RenderPassDesc {
            label: &named,
            colors: &color_attachments,
            depth: depth_attachment,
            timestamps: match (timestamps, guard_stamp) {
                (Some(i), _) => self.timer.as_ref().map(|t| wrela_gpu::PassTimestamps {
                    queries: &t.queries,
                    begin: i,
                    end: i + 1,
                }),
                (None, Some(slot)) => self.stamps.as_ref().map(|s| wrela_gpu::PassTimestamps {
                    queries: &s.queries,
                    begin: 2 * slot,
                    end: 2 * slot + 1,
                }),
                (None, None) => None,
            },
        });
        let mut prepared = prepared.iter();
        for item in &items {
            let d = match item {
                PassItem::Scissor([x, y, w, h]) => {
                    rp.set_scissor_rect(*x, *y, *w, *h);
                    continue;
                }
                PassItem::Viewport([x, y, w, h, min, max]) => {
                    rp.set_viewport(*x, *y, *w, *h, *min, *max);
                    continue;
                }
                PassItem::Draw(d) => d,
            };
            let (bind_group, offset) = prepared.next().expect("prepared for each draw");
            let Kind::Render { variants, .. } = &self.pipelines[d.pipeline as usize].kind else {
                unreachable!("the checker checked the kind");
            };
            let Some(variant) = variants.get(formats) else {
                return Err(Error::Gpu(format!(
                    "`{}` has no pipeline for its targets {formats:?}",
                    self.pipelines[d.pipeline as usize].name
                )));
            };
            rp.set_pipeline(variant);
            rp.set_bind_group(bind_group, offset.as_slice());
            counters.pipelines += 1;
            counters.bind_groups += 1;
            counters.draw_calls += 1;
            counters.draws += match d.how {
                DrawHow::Multi { count, .. } => u64::from(count),
                _ => 1,
            };
            let index = |rp: &mut wrela_gpu::RenderPass<'_>, indices: u32, at: u32| {
                rp.set_index_buffer(self.buffer(indices), u64::from(at));
            };
            match d.how {
                DrawHow::Direct { vertices, instances } => rp.draw(0..vertices, 0..instances),
                DrawHow::Indirect { buffer, offset } => {
                    rp.draw_indirect(self.buffer(buffer), u64::from(offset));
                }
                DrawHow::IndexedIndirect { indices, at, buffer, offset } => {
                    index(&mut rp, indices, at);
                    rp.draw_indexed_indirect(self.buffer(buffer), u64::from(offset));
                }
                DrawHow::Indexed {
                    indices,
                    at,
                    count,
                    instances,
                    first,
                    base_vertex,
                    first_instance,
                } => {
                    index(&mut rp, indices, at);
                    rp.draw_indexed(
                        first..first + count,
                        base_vertex,
                        first_instance..first_instance + instances,
                    );
                }
                DrawHow::Multi { indices, buffer, offset, count } => match indices {
                    Some([h, at]) => {
                        index(&mut rp, h, at);
                        rp.multi_draw_indexed_indirect(
                            self.buffer(buffer),
                            u64::from(offset),
                            count,
                        );
                    }
                    None => rp.multi_draw_indirect(self.buffer(buffer), u64::from(offset), count),
                },
            }
        }
        drop(rp);
        // A counting run's indirect draws: each one's arguments copied, after its pass, into the
        // flag's buffer, read back with the tallies (`drawn frame` lines): how many vertices and
        // instances each drew, which tools/cost prices.
        let copies: Vec<(u32, u32, u64, u64)> = match &mut self.debug_flag {
            Some(f) if f.words > f.tally => {
                let mut out = Vec::new();
                for item in &items {
                    let PassItem::Draw(d) = item else { continue };
                    let (buffer, offset, indexed) = match d.how {
                        DrawHow::Indirect { buffer, offset } => (buffer, offset, false),
                        DrawHow::IndexedIndirect { buffer, offset, .. } => (buffer, offset, true),
                        _ => continue,
                    };
                    let slot = f.drawn.len() as u64;
                    if slot >= DRAWN_MOST {
                        break;
                    }
                    f.drawn.push((d.pipeline, Arc::from(label), indexed));
                    let bytes = if indexed { 20 } else { 16 };
                    out.push((buffer, offset, bytes, 4 * (f.tally + DRAWN_WORDS * slot)));
                }
                out
            }
            _ => Vec::new(),
        };
        if let Some(f) = &self.debug_flag {
            for (buffer, offset, bytes, at) in copies {
                encoder.copy_buffer_to_buffer(
                    self.buffer(buffer),
                    u64::from(offset),
                    &f.buffer,
                    at,
                    bytes,
                );
            }
        }
        self.counters.add(&counters);
        self.encoder = Some(encoder);
        Ok(())
    }

    /// A buffer's bytes `offset..offset + size`, copied back to the CPU (waits for the GPU).
    pub(crate) fn read_range(&mut self, handle: u32, offset: u32, size: u32) -> Result<Vec<u8>> {
        self.check_debug_flag_now()?;
        self.flush()?;
        let Some(Resource::Buffer(buffer)) = self.resources.get(&handle) else {
            return Err(self.no_buffer(handle));
        };
        let staging = staging(&self.device, &mut self.buffer_readback, u64::from(size.max(4)));
        let bytes = self.copy_back(buffer, u64::from(offset), u64::from(size), &staging)?;
        self.check_errors()?;
        Ok(bytes)
    }

    /// Copies `source`'s bytes `offset..offset + size` to the CPU through `staging`, a
    /// readback buffer at least `size` bytes long, and at least 4 (waits for the GPU).
    fn copy_back(
        &self,
        source: &wrela_gpu::Buffer,
        offset: u64,
        size: u64,
        staging: &wrela_gpu::Buffer,
    ) -> Result<Vec<u8>> {
        let mut encoder = self.device.create_command_encoder("read back");
        encoder.copy_buffer_to_buffer(source, offset, staging, 0, size);
        self.device.submit(vec![encoder.finish()]);
        // A read is whole words, and at least one.
        let mapped = align_to(size, wrela_gpu::COPY_BUFFER_ALIGNMENT).clamp(4, staging.size());
        map_read_with(&self.device, staging, mapped, |bytes| bytes[..size as usize].to_vec())
    }

    /// The handles of the buffers that exist, oldest first: for tests.
    pub(crate) fn buffers(&self) -> Vec<u32> {
        let mut out: Vec<u32> = self
            .resources
            .iter()
            .filter(|(_, r)| matches!(r, Resource::Buffer(_)))
            .map(|(&h, _)| h)
            .collect();
        // Handles are given in order and never reused.
        out.sort_unstable();
        out
    }

    /// A buffer's size in bytes.
    pub(crate) fn buffer_size(&self, handle: u32) -> Result<u64> {
        match self.resources.get(&handle) {
            Some(Resource::Buffer(b)) => Ok(b.size()),
            _ => Err(self.no_buffer(handle)),
        }
    }

    /// What a command naming `handle` as a buffer finds instead: a texture or a sampler, or a
    /// handle destroyed (when), or one never made on this device.
    fn no_buffer(&self, handle: u32) -> Error {
        let what = match (self.resources.get(&handle), self.gone.get(&handle)) {
            (Some(Resource::Texture(_)), _) => "it is a texture".to_string(),
            (Some(Resource::Sampler(_)), _) => "it is a sampler".to_string(),
            (_, Some((frame, call))) => format!(
                "it was destroyed in frame {frame} (call {call}; this is frame {}, call {})",
                self.frame, self.calls
            ),
            _ => format!(
                "no buffer was made with it on this device (frame {}, call {}, device epoch {})",
                self.frame, self.calls, self.program_limits.epoch
            ),
        };
        Error::Gpu(format!("there's no buffer {handle}: {what}"))
    }

    /// Copies a whole buffer back to the CPU (waits for the GPU): for tests.
    pub(crate) fn read_buffer(&mut self, handle: u32) -> Result<Vec<u8>> {
        let size = self.buffer_size(handle)? as u32;
        self.read_range(handle, 0, size)
    }

    /// Copies the screen back: RGBA8, rows top to bottom, no padding (waits for the GPU).
    pub(crate) fn read_screen(&mut self) -> Result<Vec<u8>> {
        let screen =
            self.screen.as_ref().ok_or_else(|| Error::Gpu("there's no screen yet".into()))?;
        let (width, height) = (screen.width, screen.height);
        self.read_screen_region([0, 0], [width, height])
    }

    /// Copies the screen's pixels from `corner` (x, y), `size` (width, height) of them, back:
    /// RGBA8, rows top to bottom, no padding (waits for the GPU).
    pub(crate) fn read_screen_region(
        &mut self,
        corner: [u32; 2],
        size: [u32; 2],
    ) -> Result<Vec<u8>> {
        self.check_debug_flag_now()?;
        self.flush()?;
        let screen =
            self.screen.as_ref().ok_or_else(|| Error::Gpu("there's no screen yet".into()))?;
        let ([x, y], [width, height]) = (corner, size);
        let inside = |at: u32, n: u32, of: u32| n > 0 && at.checked_add(n).is_some_and(|e| e <= of);
        if !inside(x, width, screen.width) || !inside(y, height, screen.height) {
            return Err(Error::Gpu(format!(
                "{width}x{height} pixels at ({x}, {y}) aren't all on the {}x{} screen",
                screen.width, screen.height
            )));
        }
        let format = screen.format;
        let raw = read_rows(
            &self.device,
            &mut self.image_readback,
            &screen.texture,
            0,
            [x, y, 0],
            [width, height],
            format,
        )?;
        self.check_errors()?;
        if format == SCREEN_FORMAT {
            return Ok(raw);
        }
        // An HDR screen's half floats as RGBA8, each channel clamped to 0 to 1.
        let texel = format.bytes_per_texel() as usize;
        Ok(raw.chunks_exact(texel).flat_map(|t| as_rgba8(format, t)).collect())
    }

    /// The target of the last pass labelled `label` (`std::gpu::label`) that drew into a texture
    /// (its colour, else its depth: the level and layer it drew into), copied back as RGBA8 rows
    /// top to bottom: an 8-bit texture as it is; a float one's channels clamped to 0 to 1, one
    /// channel grey, two red and green, its alpha opaque unless it has one; a depth texture's
    /// depth grey. That level's width and height too. Waits for the GPU.
    pub(crate) fn read_target(&mut self, label: &str) -> Result<(u32, u32, Vec<u8>)> {
        let (width, height, format, raw) = self.read_target_raw(label)?;
        if format.is_uint() {
            return Err(Error::Gpu(format!(
                "the target of the pass labelled `{label}` holds integers, not an image"
            )));
        }
        let texel = format.bytes_per_texel() as usize;
        let rgba = raw.chunks_exact(texel).flat_map(|t| as_rgba8(format, t)).collect();
        Ok((width, height, rgba))
    }

    /// The labels of the passes that drew into a texture, in the order they last drew
    /// ([`Gpu::read_target`]).
    pub(crate) fn target_labels(&self) -> Vec<String> {
        self.labelled_order.iter().map(|l| l.to_string()).collect()
    }

    /// [`Gpu::read_target`]'s target as its own texels, rows top to bottom with no padding, and
    /// its format: what a comparison of two hosts compares to the bit.
    pub(crate) fn read_target_raw(
        &mut self,
        label: &str,
    ) -> Result<(u32, u32, TextureFormat, Vec<u8>)> {
        self.flush()?;
        let (handle, level, layer) = *self
            .labelled
            .get(label)
            .ok_or_else(|| Error::Gpu(format!("no pass labelled `{label}` drew into a texture")))?;
        let Some(Resource::Texture(Texture { texture, format, .. })) = self.resources.get(&handle)
        else {
            return Err(Error::Gpu(format!(
                "the target of the pass labelled `{label}` was dropped after it"
            )));
        };
        let (width, height) = stream::level_size(texture.width(), texture.height(), level);
        let format = *format;
        if format.is_compressed() {
            return Err(Error::Gpu(format!(
                "the target of the pass labelled `{label}` is compressed, which no pass draws"
            )));
        }
        let raw = read_rows(
            &self.device,
            &mut self.image_readback,
            texture,
            level,
            [0, 0, layer],
            [width, height],
            format,
        )?;
        self.check_errors()?;
        Ok((width, height, format, raw))
    }

    /// Records a copy of the flag into the next slot of its ring, in `encoder` (the frame's last
    /// work): read once it's done ([`Gpu::check_debug_flag`]).
    fn copy_flag(&mut self, encoder: &mut wrela_gpu::CommandEncoder) {
        if let Some(f) = &mut self.debug_flag {
            let slot = f.next % FLAG_RING;
            f.next += 1;
            let bytes = 4 * f.words;
            encoder.copy_buffer_to_buffer(&f.buffer, 0, &f.ring, slot * bytes, bytes);
            if let Some(z) = &f.zeros {
                encoder.copy_buffer_to_buffer(z, 0, &f.buffer, 4, bytes - 4);
            }
            f.queued = Some((slot, std::mem::take(&mut f.drawn)));
        }
    }

    /// The copy recorded by [`Gpu::copy_flag`] was submitted, in the device's latest submission.
    fn flag_submitted(&mut self) {
        let (frame, last) = (self.frame, self.device.submitted());
        if let Some(f) = &mut self.debug_flag
            && let Some((slot, drawn)) = f.queued.take()
        {
            f.pending.push_back((last, slot, frame, drawn));
        }
    }

    /// The flag now: every copy submitted read, waiting for the GPU (before a readback, so what
    /// is read back is never used as if whole when a lane ran out).
    fn check_debug_flag_now(&mut self) -> Result<()> {
        if self.debug_flag.is_none() {
            return Ok(());
        }
        let mut encoder = self.take_encoder();
        self.copy_flag(&mut encoder);
        self.encoder = Some(encoder);
        self.flush()?;
        self.flag_submitted();
        wait(&self.device);
        self.check_debug_flag()
    }

    /// Reads each copy of the flag whose frame's work is done: a lane that ran out of fuel, or
    /// (a debug build's) an index out of range, ends the run naming its pipeline.
    fn check_debug_flag(&mut self) -> Result<()> {
        let done = self.device.completed();
        let Some(f) = &mut self.debug_flag else { return Ok(()) };
        let mut found = None;
        while let Some(&(at, slot, frame, _)) = f.pending.front() {
            if at > done {
                break;
            }
            let Some((_, _, _, drawn)) = f.pending.pop_front() else { break };
            let b = f.ring.read(slot * 4 * f.words, 4 * f.words).map_err(|e| Error::Gpu(e.0))?;
            let code = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            // A counting build's tallies: each pipeline's work this frame, its low word and the
            // carries out of it (`wrela_ir::tally`).
            if f.words > 1 {
                let word = |i: usize| {
                    u64::from(u32::from_le_bytes([
                        b[4 * i],
                        b[4 * i + 1],
                        b[4 * i + 2],
                        b[4 * i + 3],
                    ]))
                };
                let per = wrela_abi::TALLY_WORDS as usize;
                for (k, p) in self.pipelines.iter().enumerate() {
                    let lo = 1 + per * k;
                    if lo + per > f.tally as usize {
                        break;
                    }
                    let work = (word(lo + 1) << 32) | word(lo);
                    let invocations = (word(lo + 3) << 32) | word(lo + 2);
                    let stream = (word(lo + 5) << 32) | word(lo + 4);
                    if work > 0 || invocations > 0 {
                        eprintln!("tally frame {frame} {} {work} {invocations} {stream}", p.name);
                    }
                }
                // Each indirect draw's vertices (an indexed draw's indices) and instances.
                for (k, (pipeline, label, _)) in drawn.iter().enumerate() {
                    let at = f.tally as usize + DRAWN_WORDS as usize * k;
                    let name = self.pipelines.get(*pipeline as usize).map_or("?", |p| &p.name);
                    eprintln!("drawn frame {frame} {label} {name} {} {}", word(at), word(at + 1));
                }
            }
            if code != 0 && found.is_none() {
                found = Some((code, frame));
            }
        }
        let Some((code, frame)) = found else { return Ok(()) };
        let fuel = code & IR_FUEL_BIT != 0;
        let number = code & !IR_FUEL_BIT;
        let (name, label) = match self.pipelines.get(number as usize - 1) {
            Some(p) => (
                format!("`{}`", p.name),
                f.labels.get(&(number - 1)).map(|l| format!(" (its pass or dispatch `{l}`)")),
            ),
            None => (format!("number {number}"), None),
        };
        let label = label.unwrap_or_default();
        if fuel {
            let most = wrela_abi::Target::HERE.limits().lane_fuel();
            return Err(Error::Gpu(format!(
                "pipeline {name}{label} ran out of fuel by frame {frame}: a lane did more than \
                 {most} work, what one does in the line's 50 ms at the GPU's warm clock \
                 (AGENTS.md, rule 2). Its loops stopped there, so what it drew or wrote is not \
                 whole. The run stops here. Bound its loops (a count the compiler can see), or \
                 cut its work over frames"
            )));
        }
        Err(Error::Gpu(format!(
            "debug build: pipeline {name} indexed an array out of range (by frame {frame}){label}; \
             the shader then reads or writes an element in range, or a zero"
        )))
    }
}

fn readback_buffer(device: &wrela_gpu::Device, size: u64) -> wrela_gpu::Buffer {
    device.create_buffer(&wrela_gpu::BufferDesc { label: "readback", size, mappable: true })
}

/// `size` (width, height) texels of `format` from `origin` (x, y, layer) of `texture`'s level
/// `level`, copied back through the readback buffer kept in `slot` (waits for the GPU): its own
/// texels (a depth texture's depth), rows top to bottom with no padding.
fn read_rows(
    device: &wrela_gpu::Device,
    slot: &mut Option<wrela_gpu::Buffer>,
    texture: &wrela_gpu::Texture,
    level: u32,
    origin: [u32; 3],
    [width, height]: [u32; 2],
    format: TextureFormat,
) -> Result<Vec<u8>> {
    let row = u64::from(width) * u64::from(format.bytes_per_texel());
    let padded = align_to(row, u64::from(wrela_gpu::COPY_BYTES_PER_ROW_ALIGNMENT));
    let size = padded * u64::from(height);
    let staging = staging(device, slot, size);
    let mut encoder = device.create_command_encoder("read a texture");
    encoder.copy_texture_to_buffer(
        wrela_gpu::TextureCopy {
            texture,
            level,
            origin,
            depth_only: format == TextureFormat::Depth32Float,
        },
        wrela_gpu::BufferCopy {
            buffer: &staging,
            offset: 0,
            bytes_per_row: padded as u32,
            rows_per_image: height,
        },
        [width, height, 1],
    );
    device.submit(vec![encoder.finish()]);
    map_read_with(device, &staging, size, |bytes| {
        let mut out = Vec::with_capacity((row * u64::from(height)) as usize);
        for padded_row in bytes.chunks_exact(padded as usize) {
            out.extend_from_slice(&padded_row[..row as usize]);
        }
        out
    })
}

/// The readback buffer kept in `slot` if it holds `size` bytes, else a new one that replaces it.
fn staging(
    device: &wrela_gpu::Device,
    slot: &mut Option<wrela_gpu::Buffer>,
    size: u64,
) -> wrela_gpu::Buffer {
    match slot {
        Some(buffer) if buffer.size() >= size => buffer.clone(),
        _ => slot.insert(readback_buffer(device, size)).clone(),
    }
}

impl Gpu {
    /// The device lost (`e` says how): a new one, every pipeline built again, every buffer,
    /// texture and sampler made again at its handle and size (zeroed: what the GPU and the
    /// program wrote is gone), the screen again at its size, and the epoch one more
    /// (`std::gpu::device_epoch`): the program writes again what it holds and builds its
    /// caches as on its first frame. A window's surface is made again on the new device at its
    /// size (the old one let go first: a window holds one swapchain), and the screen as the
    /// program last asked (HDR, Display P3) is asked again.
    fn recover(&mut self, e: &Error) -> Result<()> {
        let began = std::time::Instant::now();
        let mut fresh = Gpu::new(&self.manifest, &self.shaders, self.timing, self.report)?;
        if let Some(p) = self.presenter.take() {
            let (window, config, next_id) = (p.window.clone(), p.config, p.next_id);
            drop(p);
            fresh.attach_window(window, config.width, config.height, config.present)?;
            // The new surface's ids follow the old one's (the window's pacing reads them rising).
            if let Some(q) = &mut fresh.presenter {
                (q.id_base, q.next_id) = (next_id, next_id);
            }
        }
        let mut made: Vec<(u32, Made)> = self.made.iter().map(|(&h, &m)| (h, m)).collect();
        made.sort_by_key(|&(h, _)| h);
        for (h, m) in made {
            match m {
                Made::Buffer { size } => fresh.create_buffer(h, size),
                Made::Texture { size, format, writable, kind, levels } => {
                    fresh.create_texture(h, size, format, writable, kind, levels)
                }
                Made::Sampler { linear, repeat, compare, mip_linear, anisotropy } => {
                    fresh.create_sampler(h, linear, repeat, compare, mip_linear, anisotropy)
                }
            }
        }
        // In the plain format: a screen asked HDR becomes so again at the frame's end
        // (`show_screen`), its pipelines' variants built on the new device.
        if let Some(s) = &self.screen {
            fresh.make_screen(s.width, s.height);
        }
        let bits = self.program_limits.screen;
        if bits != 0 {
            fresh.screen_asked = Some((bits & 1 != 0, bits & 2 != 0));
        }
        fresh.frame = self.frame;
        fresh.calls = self.calls;
        fresh.labelled = std::mem::take(&mut self.labelled);
        fresh.labelled_order = std::mem::take(&mut self.labelled_order);
        fresh.asked = std::mem::take(&mut self.asked);
        // A settled read's copy was on the lost device: it says so, on the next call.
        fresh.settling = std::mem::take(&mut self.settling)
            .into_iter()
            .map(|s| Settling { lost: true, ..s })
            .collect();
        // An ask not yet answered times its label's work on the new device: the program's own
        // timer (`ask_time`) is made again there, and each ask starts at what that timer has
        // noted (the old timer's count means nothing to it).
        if fresh.timer.is_none()
            && fresh.asked.iter().any(|a| a.time.is_none())
            && fresh.device.features().has("timestamp-query")
        {
            fresh.timer = Some(Timer::new(&fresh.device, false, false));
        }
        if let Some(t) = &fresh.timer {
            for a in fresh.asked.iter_mut().filter(|a| a.time.is_none()) {
                a.from = t.noted;
            }
        }
        fresh.screen_asked = self.screen_asked.or(fresh.screen_asked);
        fresh.offscreen_hdr = self.offscreen_hdr;
        fresh.program_limits.screen = self.program_limits.screen;
        fresh.program_limits.headroom = self.program_limits.headroom;
        fresh.program_limits.epoch = self.program_limits.epoch + 1;
        fresh.submissions = self.submissions;
        // What passes cost is the scene's, not the device's.
        fresh.costs = std::mem::take(&mut self.costs);
        fresh.line_ms = self.line_ms;
        if let (Some(old), Some(new)) = (&mut self.timer, &mut fresh.timer) {
            new.results = std::mem::take(&mut old.results);
        }
        eprintln!(
            "wrela-host: {e}\nwrela-host: the device was made again at frame {} (epoch {}): {} buffers, textures and samplers, zeroed, in {:.0} ms",
            self.frame,
            fresh.program_limits.epoch,
            fresh.made.len(),
            began.elapsed().as_secs_f64() * 1000.0
        );
        *self = fresh;
        Ok(())
    }

    /// [`Gpu::end_frame`] on this device.
    fn end_frame_on_device(&mut self) -> Result<()> {
        self.record_ended()?;
        let mut encoder = self.take_encoder();
        self.indirect.end_frame(&self.device, &mut encoder);
        self.copy_flag(&mut encoder);
        self.encoder = Some(encoder);
        self.flush()?;
        self.flag_submitted();
        self.behind.submitted(&self.device);
        self.collect()?;
        self.calls += 1;
        // A loss on demand (`WRELA_GPU_LOSE_AT=<frame>`), reported as a real one is: the
        // host's handling of it is tested so.
        if self.lose_at == Some(self.frame) && self.program_limits.epoch == 0 {
            self.device.lose();
        }
        self.check_errors()?;
        self.behind.check(&self.device)?;
        self.indirect.read(&self.device)?;
        self.check_debug_flag()?;
        self.show_screen()
    }
}

impl Executor for Gpu {
    fn begin_frame(&mut self) {
        self.label = None;
    }

    fn execute(&mut self, cmd: &Command<'_>) -> Result<()> {
        // A pass that ended waits only for one that may continue it: a `Label` records it too,
        // since a pass joins only one of its own label.
        if let Some(ended) = &self.ended {
            match cmd {
                Command::BeginPass(next) if next.joins(&ended.pass) => {}
                Command::EndPass if self.pass.is_some() => {}
                c if c.opcode().in_pass() && self.pass.is_some() => {}
                _ => self.record_ended()?,
            }
        }
        match cmd {
            Command::CreateBuffer { handle, size } => self.create_buffer(*handle, *size),
            Command::CreateTexture {
                handle,
                width,
                height,
                format,
                writable,
                kind,
                layers,
                levels,
            } => self.create_texture(
                *handle,
                [*width, *height, *layers],
                *format,
                *writable,
                *kind,
                *levels,
            ),
            Command::CreateSampler { handle, linear, repeat, compare, mip_linear, anisotropy } => {
                self.create_sampler(*handle, *linear, *repeat, *compare, *mip_linear, *anisotropy)
            }
            Command::WriteBuffer { handle, offset, data } => {
                return self.write_buffer(*handle, *offset, data);
            }
            Command::WriteTexture { handle, level, layer, x, y, width, height, data } => {
                return self.write_texture(
                    *handle,
                    *level,
                    *layer,
                    [*x, *y],
                    [*width, *height],
                    data,
                );
            }
            Command::CopyTexture { source, destination } => self.copy_texture(source, destination),
            Command::CopyTextureToBuffer { source, buffer, offset, bytes_per_row } => {
                self.copy_rows(source, *buffer, *offset, *bytes_per_row, true)
            }
            Command::CopyBufferToTexture { buffer, offset, bytes_per_row, destination } => {
                self.copy_rows(destination, *buffer, *offset, *bytes_per_row, false)
            }
            Command::SetScissor { x, y, width, height } => {
                self.pass_item(PassItem::Scissor([*x, *y, *width, *height]))
            }
            // Done between frames ([`Gpu::show_screen`]).
            Command::ConfigureScreen { hdr, gamut } => {
                self.screen_asked = Some((*hdr, *gamut == 1));
            }
            Command::SetViewport { rect: [x, y, w, h], min_depth, max_depth } => {
                self.pass_item(PassItem::Viewport([*x, *y, *w, *h, *min_depth, *max_depth]))
            }
            Command::DrawIndexed {
                pipeline,
                indices,
                index_offset,
                index_count,
                instances,
                first_index,
                base_vertex,
                first_instance,
                bindings,
                uniforms,
                ..
            } => {
                // Its ranges' ends are `u32`s, as a draw takes them.
                if first_instance.checked_add(*instances).is_none() {
                    return Err(Error::command(
                        cmd.opcode(),
                        format!(
                            "{instances} instances from {first_instance} run past instance {}",
                            u32::MAX
                        ),
                    ));
                }
                if first_index.checked_add(*index_count).is_none() {
                    return Err(Error::command(
                        cmd.opcode(),
                        format!(
                            "{index_count} indices from {first_index} run past index {}",
                            u32::MAX
                        ),
                    ));
                }
                let how = DrawHow::Indexed {
                    indices: *indices,
                    at: *index_offset,
                    count: *index_count,
                    instances: *instances,
                    first: *first_index,
                    base_vertex: *base_vertex,
                    first_instance: *first_instance,
                };
                self.draw(*pipeline, how, bindings, uniforms);
            }
            Command::MultiDrawIndirect {
                pipeline,
                indices,
                arguments,
                offset,
                count,
                count_buffer,
                bindings,
                uniforms,
                ..
            } => {
                if *count_buffer != stream::NONE {
                    return Err(Error::command(
                        cmd.opcode(),
                        "a draw count read from a buffer isn't supported: give the count",
                    ));
                }
                let how = DrawHow::Multi {
                    indices: indices.map(|[h, at, _]| [h, at]),
                    buffer: *arguments,
                    offset: *offset,
                    count: *count,
                };
                self.draw(*pipeline, how, bindings, uniforms);
            }
            Command::CopyBuffer {
                source,
                source_offset,
                destination,
                destination_offset,
                size,
            } => {
                self.copy_buffer(*source, *source_offset, *destination, *destination_offset, *size)
            }
            Command::Dispatch { pipeline, groups, bindings, uniforms } => {
                return self.dispatch(*pipeline, Ok(*groups), bindings, uniforms);
            }
            Command::DispatchIndirect { pipeline, arguments, offset, most, bindings, uniforms } => {
                return self.dispatch(
                    *pipeline,
                    Err((*arguments, *offset, *most)),
                    bindings,
                    uniforms,
                );
            }
            Command::BeginScreenPass { clear } => {
                if !self.label_fresh {
                    self.label = None;
                }
                self.begin_pass(Pass::one(stream::SCREEN, false, *clear))
            }
            // Its room in the submission is made where it's recorded (`Gpu::record_piece`).
            Command::BeginPass(pass) => self.begin_pass(*pass),
            Command::Draw { pipeline, vertices, instances, bindings, uniforms } => {
                let how = DrawHow::Direct { vertices: *vertices, instances: *instances };
                self.draw(*pipeline, how, bindings, uniforms);
            }
            Command::DrawIndirect { pipeline, arguments, offset, bindings, uniforms } => {
                let how = DrawHow::Indirect { buffer: *arguments, offset: *offset };
                self.draw(*pipeline, how, bindings, uniforms);
            }
            Command::DrawIndexedIndirect {
                pipeline,
                indices,
                index_offset,
                arguments,
                offset,
                bindings,
                uniforms,
                ..
            } => {
                let how = DrawHow::IndexedIndirect {
                    indices: *indices,
                    at: *index_offset,
                    buffer: *arguments,
                    offset: *offset,
                };
                self.draw(*pipeline, how, bindings, uniforms);
            }
            Command::Present => return self.end_pass(Opcode::Present),
            Command::EndPass => return self.end_pass(Opcode::EndPass),
            Command::DestroyBuffer { handle }
            | Command::DestroyTexture { handle }
            | Command::DestroySampler { handle } => self.destroy(*handle),
            // Requests are the program's host's business (`program`), not the GPU's.
            Command::ReadBuffer { .. }
            | Command::ReadSettled { .. }
            | Command::StorageRead { .. }
            | Command::StorageWrite { .. }
            | Command::Fetch { .. }
            | Command::Post { .. }
            | Command::HostRequest { .. }
            | Command::Prepare { .. }
            | Command::Unprepare { .. }
            | Command::HandOver { .. }
            | Command::Timed { .. } => {}
            // The passes and dispatches after it share it; a name given again is kept.
            Command::Label { name } => {
                if self.label.as_deref() != Some(*name) {
                    self.label = Some(Arc::from(*name));
                }
                self.label_fresh = true;
            }
        }
        Ok(())
    }

    fn end_frame(&mut self) -> Result<()> {
        match self.end_frame_on_device() {
            Err(e) if says_lost(&e) => self.recover(&e),
            done => done,
        }
    }

    /// A program's ask for `label`'s time (`std::gpu::timed`): from it on, until it's answered,
    /// the passes and dispatches labelled `label` are timed, and nothing else is (outside a run
    /// timed from its start), so a frame that times nothing doesn't wait for its timestamps.
    /// Its answer is that work's alone, in the first call that has some: what the program
    /// recorded before the ask doesn't count, even in the same call.
    fn ask_time(&mut self, request: u32, label: &str) -> std::result::Result<(), String> {
        let timer = match &mut self.timer {
            Some(t) => t,
            None => {
                if !self.device.features().has("timestamp-query") {
                    return Err("this GPU can't time its work: it has no timestamp queries".into());
                }
                self.timer.insert(Timer::new(&self.device, false, false))
            }
        };
        let ask = Ask { request, label: label.to_string(), from: timer.noted, time: None };
        self.asked.push(ask);
        Ok(())
    }

    fn times(&mut self) -> Vec<(u32, std::result::Result<Vec<u8>, String>)> {
        let mut answered = Vec::new();
        // A call's work is all read back once it has ended.
        let ended = self.calls;
        self.asked.retain(|ask| match ask.time {
            Some((call, start, end)) if call < ended => {
                let nanos = (end - start).max(0.0);
                answered.push((ask.request, Ok(((nanos / 1000.0) as f32).to_le_bytes().to_vec())));
                false
            }
            _ => true,
        });
        answered
    }

    fn read_back(&mut self, handle: u32, offset: u32, size: u32) -> Result<Vec<u8>> {
        self.read_range(handle, offset, size)
    }

    /// Copies the range, after the work recorded so far, into a buffer of its own, and answers
    /// once its submission is done ([`Executor::settled`]): nothing waits for the GPU.
    fn read_settled(
        &mut self,
        request: u32,
        handle: u32,
        offset: u32,
        size: u32,
    ) -> Option<Result<Vec<u8>>> {
        if let Err(e) = self.record_ended() {
            return Some(Err(e));
        }
        let Some(Resource::Buffer(buffer)) = self.resources.get(&handle) else {
            return Some(Err(self.no_buffer(handle)));
        };
        let buffer = buffer.clone();
        let staging = readback_buffer(&self.device, u64::from(size.max(4)));
        let mut encoder = self.take_encoder();
        encoder.copy_buffer_to_buffer(&buffer, u64::from(offset), &staging, 0, u64::from(size));
        self.encoder = Some(encoder);
        self.settling.push(Settling {
            request,
            staging,
            size: u64::from(size),
            done_at: None,
            lost: false,
        });
        None
    }

    fn settled(&mut self) -> Vec<(u32, std::result::Result<Vec<u8>, String>)> {
        let done = self.device.completed();
        let mut answered = Vec::new();
        self.settling.retain(|s| {
            let answer = if s.lost {
                Err("the device was lost before the read's work was done".to_string())
            } else {
                match s.done_at {
                    Some(at) if at <= done => {
                        // Its submission is done: mapping it doesn't wait.
                        let mapped = align_to(s.size, wrela_gpu::COPY_BUFFER_ALIGNMENT)
                            .clamp(4, s.staging.size());
                        s.staging
                            .read(0, mapped)
                            .map(|bytes| bytes[..s.size as usize].to_vec())
                            .map_err(|e| e.0)
                    }
                    _ => return true,
                }
            };
            answered.push((s.request, answer));
            false
        });
        answered
    }

    fn limits(&self) -> wrela_abi::Limits {
        self.program_limits
    }

    /// Finishes the old build's work, forgets its resources, and takes the new build's
    /// pipelines. The device, the screen and the rings stay.
    fn reload(&mut self, manifest: &Manifest, shaders: &[Vec<u8>]) -> Result<()> {
        self.finish_build()?;
        self.take_build(manifest, shaders)
    }
}

/// Makes a render pipeline's variant for `targets`, if it isn't made yet.
fn render_variant(
    device: &wrela_gpu::Device,
    name: &str,
    layout: &wrela_gpu::BindGroupLayout,
    kind: &mut Kind,
    targets: Targets,
) -> std::result::Result<(), String> {
    let Kind::Render { module, vertex, fragment, blends, cull, depth_bias, depth, variants } = kind
    else {
        return Ok(());
    };
    if variants.contains_key(&targets) {
        return Ok(());
    }
    let colors: Vec<wrela_gpu::ColorTarget> = targets
        .0
        .iter()
        .enumerate()
        .map(|(c, &format)| wrela_gpu::ColorTarget {
            format,
            blend: blends.get(c).copied().unwrap_or_default(),
        })
        .collect();
    let depth =
        targets.1.map(|format| wrela_gpu::DepthDesc { format, state: *depth, bias: *depth_bias });
    // Triangle list, counter-clockwise front faces: WebGPU's defaults.
    let pipeline = device
        .create_render_pipeline(&wrela_gpu::RenderPipelineDesc {
            label: name,
            layout,
            module,
            vertex_entry: vertex,
            fragment_entry: fragment,
            colors: &colors,
            depth,
            cull: *cull,
        })
        .map_err(|e| e.0)?;
    variants.insert(targets, pipeline);
    Ok(())
}

/// A mappable buffer's first `len` bytes (whole words), converted by `read` in place, once the
/// GPU is done with what's submitted: callers copy just the pixels or words they need.
fn map_read_with<T>(
    device: &wrela_gpu::Device,
    buffer: &wrela_gpu::Buffer,
    len: u64,
    read: impl FnOnce(&[u8]) -> T,
) -> Result<T> {
    wait(device);
    let bytes = buffer.read(0, len).map_err(|e| Error::Gpu(e.0))?;
    Ok(read(&bytes))
}

impl Timer {
    fn new(device: &wrela_gpu::Device, serial: bool, keep: bool) -> Timer {
        let (resolved, readback) = query_buffers(device, TIMESTAMP_QUERIES);
        Timer {
            serial,
            keep,
            noted: 0,
            queries: device
                .create_query_set(TIMESTAMP_QUERIES)
                .expect("a device with timestamp queries makes a query set"),
            capacity: TIMESTAMP_QUERIES,
            resolved,
            readback,
            pending: Vec::new(),
            results: Vec::new(),
            submissions: 0,
        }
    }

    /// Twice the queries, up to `TIMESTAMP_QUERIES_MOST`, once every query is read (`pending`
    /// empty); as it was where the device makes no larger set.
    fn grow(&mut self, device: &wrela_gpu::Device) {
        let want = (self.capacity * 2).min(TIMESTAMP_QUERIES_MOST);
        if want <= self.capacity || !self.pending.is_empty() {
            return;
        }
        if let Ok(queries) = device.create_query_set(want) {
            (self.resolved, self.readback) = query_buffers(device, want);
            self.queries = queries;
            self.capacity = want;
        }
    }

    /// Notes a pass or dispatch about to be timed, if it is, and returns the first of its two
    /// query indices: a timed run times every one; a program's asks (`asked`, `ask_time`) only
    /// their labels'. [`Gpu::reserve_timestamps`] must have made room.
    fn next_if(&mut self, asked: &[Ask], frame: usize, call: usize, label: &str) -> Option<u32> {
        if !self.keep && !asked.iter().any(|a| a.label == label) {
            return None;
        }
        self.pending.push(Timed { frame, call, label: label.to_string(), at: self.noted });
        self.noted += 1;
        Some((self.pending.len() as u32 - 1) * 2)
    }

    /// Reads back the timestamps of the submissions made since the last call, adding each to
    /// the answers of the asks before it. The span mode calls it at every flush, so it measures
    /// GPU durations, not throughput; the serial mode, which waits after each pass anyway, once
    /// a frame.
    fn collect(&mut self, device: &wrela_gpu::Device, asked: &mut [Ask]) -> Result<()> {
        let times = read_timestamp_spans(
            device,
            &self.queries,
            &self.resolved,
            &self.readback,
            self.pending.len() as u32,
        )?;
        for (timed, (start, end)) in self.pending.drain(..).zip(times) {
            let Timed { frame, call, label, at } = timed;
            let submission = self.submissions;
            let nanos = end - start;
            for ask in asked.iter_mut().filter(|a| a.label == label && a.from <= at) {
                match &mut ask.time {
                    None => ask.time = Some((call, start, end)),
                    Some((c, s, e)) if *c == call => {
                        *s = s.min(start);
                        *e = e.max(end);
                    }
                    Some(_) => {}
                }
            }
            if self.keep {
                self.results.push(GpuTiming { frame, label, nanos, start, end, submission });
            }
            if self.serial {
                self.submissions += 1;
            }
        }
        if !self.serial {
            self.submissions += 1;
        }
        Ok(())
    }
}

/// The starts and ends of `passes` timed passes, in nanoseconds on the GPU's clock (an end before
/// its start, as a clock that wrapped, is taken as the start). Pass `i` wrote queries `2i`
/// (start) and `2i + 1` (end) of `queries`, in a submission already made. Waits for the GPU.
///
/// The queries are resolved in a later submission, after the timed one completes: on Apple GPUs
/// a resolve in the same command buffer as the passes it times can read the previous
/// submission's values.
fn read_timestamp_spans(
    device: &wrela_gpu::Device,
    queries: &wrela_gpu::QuerySet,
    resolved: &wrela_gpu::Buffer,
    readback: &wrela_gpu::Buffer,
    passes: u32,
) -> Result<Vec<(f64, f64)>> {
    let ticks = resolve(device, queries, resolved, readback, 0, passes * 2)?;
    let period = f64::from(device.timestamp_period());
    Ok(ticks
        .chunks_exact(2)
        .map(|pair| (pair[0] as f64 * period, pair[0].max(pair[1]) as f64 * period))
        .collect())
}

/// Storage for resolving and reading back `n` timestamp queries.
fn query_buffers(device: &wrela_gpu::Device, n: u32) -> (wrela_gpu::Buffer, wrela_gpu::Buffer) {
    let size = u64::from(n) * 8;
    let resolved =
        device.create_buffer(&wrela_gpu::BufferDesc { label: "queries", size, mappable: false });
    let readback = device.create_buffer(&wrela_gpu::BufferDesc {
        label: "queries readback",
        size,
        mappable: true,
    });
    (resolved, readback)
}

/// The values of `n` timestamp queries from `first`, in a submission already made. Waits for
/// the GPU, then resolves them and reads them back.
fn resolve(
    device: &wrela_gpu::Device,
    queries: &wrela_gpu::QuerySet,
    resolved: &wrela_gpu::Buffer,
    readback: &wrela_gpu::Buffer,
    first: u32,
    n: u32,
) -> Result<Vec<u64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    wait(device);
    let size = u64::from(n) * 8;
    let mut encoder = device.create_command_encoder("resolve timestamps");
    encoder.resolve_query_set(queries, first, n, resolved, 0);
    encoder.copy_buffer_to_buffer(resolved, 0, readback, 0, size);
    device.submit(vec![encoder.finish()]);
    map_read_with(device, readback, size, |bytes| {
        bytes.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes"))).collect()
    })
}

/// A piece's own time (ms) from the guard's stamps: its pass's begin and end (slot `slot`, a
/// submission already done), on the GPU's clock. None where the stamps don't read as a time.
fn stamped_ms(device: &wrela_gpu::Device, stamps: &Stamps, slot: u32) -> Option<f64> {
    let Stamps { queries, resolved, readback } = stamps;
    let ticks = resolve(device, queries, resolved, readback, 2 * slot, 2).ok()?;
    let period = f64::from(device.timestamp_period());
    let (begin, end) = (ticks[0], ticks[1]);
    (begin > 0 && end >= begin).then(|| (end - begin) as f64 * period / 1.0e6)
}
