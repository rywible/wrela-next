//! # The wrela runtime ABI
//!
//! This crate is the one definition of how a compiled wrela program talks to a host (the native
//! host in `toolchain/host`). It's an executable spec: the format's version, opcodes and
//! layouts are defined here, documented here, and pinned by golden byte tests. Std's constants
//! are generated from it ([`wrela`]); a test fails if a checked-in copy drifts.
//!
//! ## What a build ships (D-099)
//!
//! - `game.wasm`: the program's CPU code.
//! - One WGSL module per pipeline.
//! - `manifest.json`: a [`Manifest`]. It names the WASM and lists every pipeline.
//! - The standard runtime, pinned to the version the program was built with (D-100).
//!
//! ## The program ABI
//!
//! The WASM module
//! - exports its linear memory as `memory`;
//! - exports `frame(time: f32, width: u32, height: u32)`, which the host calls once per frame
//!   with the seconds since the program started and the canvas size in pixels;
//! - may export `init()`, which the host calls once, after instantiating the module and before
//!   anything else: a program with state makes it there (language.md §15);
//! - imports `wrela.submit(ptr: i32, len: i32)`, which hands the host a batch of commands: `len`
//!   bytes at `ptr` in its memory (see [`stream`]); and may import
//!   `wrela.request_status(request: i32) -> i32` and `wrela.request_take(request: i32, ptr: i32)`,
//!   which answer the requests its batches make, `wrela.limit(index: i32) -> i32`, which
//!   gives one of the GPU's limits ([`Limits`]), `wrela.audio(task: i32, context: i32)`,
//!   which starts the program's voice ([`IMPORT_AUDIO`]), `wrela.tick(task: i32, context: i32,
//!   hz: i32)`, which starts its ticker ([`IMPORT_TICK`]), and `wrela.input(ptr: i32, cap: i32)
//!   -> i32`, which gives it the pointer and keyboard events that have arrived ([`input`]);
//! - imports its memory, shared, as `wrela.memory` ([`memory`]);
//! - may export thread entries (`__worker`, `__audio`, `__tick`), which a host calls on the
//!   program's other threads ([`memory`]'s threads);
//! - may have a start function, which runs as each instance starts and may not call the host:
//!   the compiler's copies the program's constants into the memory, once however many instances
//!   share it ([`memory::DATA_READY`]).
//!
//! Nothing else is imported, so nothing host-dependent can reach the program's arithmetic
//! (D-015). Other exports are the program's own `pub fn`s, which tests may call.
//!
//! A tool build (language.md §15.5) exports `tool` ([`EXPORT_TOOL`]) in place of `frame`, and
//! `__tool_worker` in place of `__worker`; it imports [`TOOL_FUNCTIONS`], which no other build
//! does, and makes its requests through them, not through `submit`.
//!
//! ## The state hash
//!
//! [`hash::StateHash`] is FNV-1a 64 over every byte of every batch the program submits, in
//! order. Two hosts running the same program at the same frame times must compute the same
//! hash (AC7).

pub mod check;
pub mod hash;
pub mod input;
pub mod lines;
pub mod manifest;
pub mod memory;
pub mod metal;
pub mod stream;
pub mod table;
pub mod ticks;
pub mod vectors;
pub mod vulkan;
pub mod wrela;

pub use input::{IMPORT_INPUT, IMPORT_PAYLOAD};
pub use manifest::Manifest;

use std::path::{Path, PathBuf};

/// The program's imports: their module, and each one's name. `submit` hands the host a batch;
/// `request_status` and `request_take` answer requests ([`stream`]).
pub const IMPORT_MODULE: &str = "wrela";
pub const IMPORT_SUBMIT: &str = "submit";
pub const IMPORT_REQUEST_STATUS: &str = "request_status";
pub const IMPORT_REQUEST_TAKE: &str = "request_take";
pub const IMPORT_LIMIT: &str = "limit";
/// `wrela.audio(task, context)`: starts the program's voice, which the audio thread renders by
/// calling [`EXPORT_AUDIO`] with the same two numbers (language.md §16.4). Once per program.
pub const IMPORT_AUDIO: &str = "audio";
/// `wrela.tick(task, context, hz)`: starts the program's ticker, which the ticker's thread
/// steps `hz` times a second by calling [`EXPORT_TICK`] with the two numbers and the tick's
/// (language.md's `std::tick`). Once per program: a second is a panic.
pub const IMPORT_TICK: &str = "tick";
/// The memory, which the module imports, shared (`memory`), and also exports.
pub const IMPORT_MEMORY: &str = "memory";
/// The same memory, as the module exports it.
pub const EXPORT_MEMORY: &str = "memory";
/// `__worker(thread)`: a helper's thread entry (`memory`'s threads and helpers).
pub const EXPORT_WORKER: &str = "__worker";
/// `__audio(thread, task, context)`: renders one quantum of the voice `wrela.audio` started, at
/// [`memory::AUDIO_OUT`] (its left channel, then its right), on the audio thread
/// ([`memory::THREAD_AUDIO`]), on its own instance of the module, with the same memory.
pub const EXPORT_AUDIO: &str = "__audio";
/// `__tick(thread, task, context, tick)`: runs tick `tick` of the ticker `wrela.tick` started,
/// with the records at [`memory::TICK_RECORDS`], on the ticker's thread
/// ([`memory::THREAD_TICK`]), on its own instance of the module, with the same memory.
pub const EXPORT_TICK: &str = "__tick";
/// `__lift_set(literal, value)`: in a lifted build (language.md §23), changes a literal's value,
/// as `std::lift::set` does: code reads the new one from its next call, on the CPU and the GPU.
/// A host calls it on the program's thread, between calls (a tool's edit, hot reload).
pub const EXPORT_LIFT_SET: &str = "__lift_set";
/// `keep(address, length)`: the host keeps those bytes of the program's memory for the build
/// that replaces this one while it runs (hot reload, `std::reload`), in place of what it kept
/// before. A host that starts a program anew keeps nothing.
pub const IMPORT_KEEP: &str = "keep";
/// `kept(address, cap) -> length`: copies up to `cap` bytes of what the replaced build last
/// kept to the address, and gives how many it kept, all of them (0 if nothing). The tool host
/// gives a tool its start this way instead: its arguments and granted secrets (`std::tool`).
pub const IMPORT_KEPT: &str = "kept";
/// `clock(address)`: writes the seconds since the host started the program, an `f64` that never
/// goes back, to the address (`std::time::now`). Reading it is `nondet` (language.md §8).
pub const IMPORT_CLOCK: &str = "clock";
/// `print(address, length)`: shows that UTF-8 line on the host's console now, from any thread
/// (`std::io::print`). The program never reads it back (the `trace` effect, language.md §8).
pub const IMPORT_PRINT: &str = "print";
/// `phase(address, length)`: this thread's time from now to its next `phase`, or to the end of
/// the call the host made (a frame, a tick, a constant, a test), counts under that UTF-8 name;
/// an empty name counts it under none (`std::time::phase`). The host reports it, and the
/// program never reads it back (the `trace` effect).
pub const IMPORT_PHASE: &str = "phase";
/// The audio thread's sample rate, in hertz, and how many samples one `__audio` call renders
/// (a Web Audio render quantum). Both hosts render at this rate; the browser resamples to the
/// device's.
pub const AUDIO_SAMPLE_RATE: u32 = 48000;
pub const AUDIO_QUANTUM: u32 = 128;
/// The voice's channels: left, then right.
pub const AUDIO_CHANNELS: u32 = 2;
/// What `request_status` says of a request that isn't answered yet, of one that failed, and of a
/// storage read of a path that holds nothing (nothing was stored there), which failed too.
pub const REQUEST_PENDING: i32 = -1;
pub const REQUEST_FAILED: i32 = -2;
pub const REQUEST_MISSING: i32 = -3;

/// A function the host gives a program, in [`IMPORT_MODULE`]: its name, and how many `i32`s it
/// takes and gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostFunction {
    pub name: &'static str,
    pub params: usize,
    pub results: usize,
}

impl HostFunction {
    /// Its type, as both hosts word it: `(i32, i32) -> ()`.
    pub fn signature(&self) -> String {
        let list = |n| vec!["i32"; n].join(", ");
        format!("({}) -> ({})", list(self.params), list(self.results))
    }
}

/// Every function a program may import besides its memory ([`IMPORT_MEMORY`]).
pub const HOST_FUNCTIONS: [HostFunction; 13] = [
    HostFunction { name: IMPORT_SUBMIT, params: 2, results: 0 },
    HostFunction { name: IMPORT_REQUEST_STATUS, params: 1, results: 1 },
    HostFunction { name: IMPORT_REQUEST_TAKE, params: 2, results: 0 },
    HostFunction { name: IMPORT_LIMIT, params: 1, results: 1 },
    HostFunction { name: IMPORT_AUDIO, params: 2, results: 0 },
    HostFunction { name: IMPORT_INPUT, params: 2, results: 1 },
    HostFunction { name: IMPORT_PAYLOAD, params: 4, results: 1 },
    HostFunction { name: IMPORT_TICK, params: 3, results: 0 },
    HostFunction { name: IMPORT_KEEP, params: 2, results: 0 },
    HostFunction { name: IMPORT_KEPT, params: 2, results: 1 },
    HostFunction { name: IMPORT_CLOCK, params: 1, results: 0 },
    HostFunction { name: IMPORT_PRINT, params: 2, results: 0 },
    HostFunction { name: IMPORT_PHASE, params: 2, results: 0 },
];

/// The export the host calls each frame.
pub const EXPORT_FRAME: &str = "frame";
/// The export a program with state has (language.md §15): the host calls it once, after
/// instantiating the program and before any other export.
pub const EXPORT_INIT: &str = "init";
/// A server build's exports (language.md §15): the server host calls `serve_init` once, when it
/// starts the program, and `serve` whenever something happens (an event, a timer).
pub const EXPORT_SERVE_INIT: &str = "serve_init";
pub const EXPORT_SERVE: &str = "serve";
/// A tool build's export (language.md §15.3, §15.5): `tool() -> u32`, which the tool host calls
/// once, on the tool's thread. It runs the whole tool, waiting on its requests in the host
/// ([`TOOL_FUNCTIONS`]), and its result is the tool's exit status (0 to 255).
pub const EXPORT_TOOL: &str = "tool";
/// `__tool_worker(thread)`: a tool build's helpers' thread entry, in place of
/// [`EXPORT_WORKER`]: the same loop, which tells the host when the helper waits for work and
/// when it wakes ([`IMPORT_TOOL_IDLE`], [`IMPORT_TOOL_ACTIVE`]).
pub const EXPORT_TOOL_WORKER: &str = "__tool_worker";
/// `tool_request(service_at, service_len, body_at, body_len) -> request`: makes a request of
/// the tool service named by that UTF-8 (`tool/files/read`), with those bytes, at once, and
/// gives the number the host chose for it. Any thread of a tool build may call it. The tool
/// host carries the request out beside the tool (language.md §15.5).
pub const IMPORT_TOOL_REQUEST: &str = "tool_request";
/// `tool_wait(request) -> length`: blocks the calling thread until the request is answered,
/// and gives the answer's length in bytes, or [`REQUEST_FAILED`] if the host has no such
/// service. If the run must stop while it waits (its timeout, an interrupt, a helper's trap,
/// stuck work), it traps the calling thread instead, with that reason.
pub const IMPORT_TOOL_WAIT: &str = "tool_wait";
/// `tool_any(at, count) -> index`: blocks until one of the `count` requests whose numbers
/// (`u32`s) are at `at` is answered, and gives its index there; it reads none of them. It
/// traps as [`IMPORT_TOOL_WAIT`] does.
pub const IMPORT_TOOL_ANY: &str = "tool_any";
/// `tool_take(request, at)`: copies an answered request's answer to `at` and forgets the
/// request. What the request made (blobs, driven builds, servers, browsers) is the tool's from
/// here on.
pub const IMPORT_TOOL_TAKE: &str = "tool_take";
/// `tool_idle()`: a helper is about to wait for work, holding none ([`EXPORT_TOOL_WORKER`]).
/// A tool test's fake clock reads it (language.md §13.2).
pub const IMPORT_TOOL_IDLE: &str = "tool_idle";
/// `tool_active()`: a helper woke, before it looks for work.
pub const IMPORT_TOOL_ACTIVE: &str = "tool_active";
/// `tool_join(slot)`: blocks the calling thread while the task in slot `slot` (its address) is
/// [`memory::SLOT_STARTED`] on another thread, as a tool task's join (`std::tool::Task`). It
/// traps as [`IMPORT_TOOL_WAIT`] does.
pub const IMPORT_TOOL_JOIN: &str = "tool_join";
/// `tool_done()`: a helper finished a task and wrote its slot: a thread in `tool_join` may go
/// on.
pub const IMPORT_TOOL_DONE: &str = "tool_done";

/// The functions only a tool build imports, besides [`HOST_FUNCTIONS`]: a tool's requests, and
/// what its threads tell the host. A tool build makes no request through `submit` (the tool
/// host refuses `submit`, `request_status` and `request_take`), and no other build imports
/// these.
pub const TOOL_FUNCTIONS: [HostFunction; 8] = [
    HostFunction { name: IMPORT_TOOL_REQUEST, params: 4, results: 1 },
    HostFunction { name: IMPORT_TOOL_WAIT, params: 1, results: 1 },
    HostFunction { name: IMPORT_TOOL_ANY, params: 2, results: 1 },
    HostFunction { name: IMPORT_TOOL_TAKE, params: 2, results: 0 },
    HostFunction { name: IMPORT_TOOL_IDLE, params: 0, results: 0 },
    HostFunction { name: IMPORT_TOOL_ACTIVE, params: 0, results: 0 },
    HostFunction { name: IMPORT_TOOL_JOIN, params: 1, results: 0 },
    HostFunction { name: IMPORT_TOOL_DONE, params: 0, results: 0 },
];
/// The custom section both of a program's builds carry (language.md §15.3): eight bytes,
/// little-endian, the [`pair_id`] of its client and server modules, which a client and a server
/// compare to know they're one program's.
pub const PAIR_SECTION: &str = "wrela.pair";

/// The pair id of a program's modules, each without its [`PAIR_SECTION`]: FNV-1a 64 over each
/// one's length (eight bytes, little-endian, 0 for a role the program hasn't) and bytes, the
/// client's first.
pub fn pair_id(client: Option<&[u8]>, server: Option<&[u8]>) -> u64 {
    let mut hash = crate::hash::StateHash::new();
    for wasm in [client, server] {
        let bytes = wasm.unwrap_or(&[]);
        hash.update(&(bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    hash.value()
}

/// The pair id in a module's [`PAIR_SECTION`], if it has one: the module's sections are read
/// in order, and anything malformed gives `None`.
pub fn read_pair(wasm: &[u8]) -> Option<u64> {
    if wasm.get(..8)? != b"\0asm\x01\0\0\0" {
        return None;
    }
    let mut pair = None;
    for section in lines::sections(wasm) {
        let (id, body) = section?;
        if id != 0 {
            continue;
        }
        let (name, payload) = lines::custom(body)?;
        if name == PAIR_SECTION.as_bytes() {
            pair.get_or_insert(u64::from_le_bytes(payload.try_into().ok()?));
        }
    }
    pair
}

/// The machine a build is for, which decides its shaders' language and the limits it is compiled
/// for: Apple's GPUs through Metal (Metal shading language), or everything else through Vulkan
/// (SPIR-V). A build runs on its own target's host alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    Metal,
    Vulkan,
}

impl Target {
    /// Both, in the order a list of them is written.
    pub const ALL: [Target; 2] = [Target::Metal, Target::Vulkan];

    /// The machine this program runs on: Metal on Apple's, Vulkan on any other.
    pub const HERE: Target =
        if cfg!(target_vendor = "apple") { Target::Metal } else { Target::Vulkan };

    /// Its name in a manifest and on a command line.
    pub fn name(self) -> &'static str {
        match self {
            Target::Metal => "metal",
            Target::Vulkan => "vulkan",
        }
    }

    /// The target named `name`, if there is one.
    pub fn parse(name: &str) -> Option<Target> {
        Target::ALL.into_iter().find(|t| t.name() == name)
    }

    /// The limits a pipeline for it is compiled for, which its host requires of a device.
    pub const fn limits(self) -> &'static TargetLimits {
        match self {
            Target::Metal => &TargetLimits::METAL,
            Target::Vulkan => &TargetLimits::VULKAN,
        }
    }
}

impl Default for Target {
    fn default() -> Target {
        Target::HERE
    }
}

impl serde::Serialize for Target {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

/// What a target's pipelines are compiled for and its commands checked against: the device's
/// limits a program sees ([`Limits`]), and what one shader stage may bind and pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetLimits {
    pub device: Limits,
    /// Storage buffers one stage binds (`maxStorageBuffersPerShaderStage`).
    pub storage_buffers_per_stage: usize,
    /// Sampled textures, samplers and storage textures one stage binds.
    pub sampled_textures_per_stage: usize,
    pub samplers_per_stage: usize,
    pub storage_textures_per_stage: usize,
    /// The values a vertex shader passes to a fragment shader (`maxInterStageShaderVariables`):
    /// one each, a matrix one a column.
    pub varyings: usize,
    /// The bytes a sample of a pass's colour targets take (`maxColorAttachmentBytesPerSample`).
    pub color_attachment_bytes: u32,
    /// The buffer slots of a stage's argument table, where the target has one (Metal's 31: the
    /// storage buffers, the uniform block, a debug build's flag and the buffers' sizes together;
    /// in a kernel that reaches the device's [`table`], two fewer).
    pub buffer_slots_per_stage: Option<usize>,
    /// The most time one unit of a kernel's work bound (language.md §18.11) takes on the
    /// machine's GPU, in femtoseconds: a dispatch's time over its bound times its invocations,
    /// measured on kernels that do all their bound's arithmetic (toolchain/tests/steps/calibrate;
    /// docs/rates.md, "The limit"). Reads that miss every cache take up to 100 times as long a
    /// unit: a kernel made of them can run past what this says.
    pub step_femtoseconds: u32,
    /// How many invocations the GPU runs side by side, about: a dispatch of fewer takes as long
    /// as one of this many, since one invocation alone does its bound's work in a row.
    pub step_width: u32,
    /// How long the reference machine takes for the host's calibration kernel (a chain of sines,
    /// `wrela-host`'s `Gpu::calibration_ns`), in nanoseconds: a device's own time over this
    /// scales [`TargetLimits::step_femtoseconds`] there (0: no calibration).
    pub calibration_ns: u32,
    /// How many tenths of its warm time a lane's work may take at the clock this GPU lowers to
    /// when it sits idle: the most a pass ran slower after an idle stretch, measured
    /// (`wrela-host`'s `how_much_a_pass_slows_after_idle`). A lane's fuel is what it does in
    /// the line at that clock ([`TargetLimits::lane_fuel`]).
    pub idle_slowdown_tenths: u32,
}

impl TargetLimits {
    /// The most work one lane (one invocation) does in [`STEP_MS`] at the GPU's warm clock: one
    /// invocation alone at [`TargetLimits::step_femtoseconds`] times
    /// [`TargetLimits::step_width`] a unit (a lane is a small share of the GPU: the unit is the
    /// whole GPU's a unit of work, so one lane alone takes `step_width` units' time a unit).
    /// No lane of a draw does more: a draw whose lanes' bound is within it is left as written,
    /// and one with a loop that could pass it is charged against it ([`TargetLimits::lane_fuel`]).
    /// At the idle clock such a lane holds its tile at most
    /// [`TargetLimits::idle_slowdown_tenths`] / 10 times the line, which the guard ends after the
    /// fact; it can't hang the GPU.
    pub fn lane_line_work(&self) -> u64 {
        let unit_fs = u64::from(self.step_femtoseconds.max(1)) * u64::from(self.step_width.max(1));
        u64::from(STEP_MS) * 1_000_000_000_000 / unit_fs
    }

    /// A lane's fuel (language.md §18.11): the most work one lane may do where a loop has no
    /// bound or its bound is past it, which such a loop is charged against pass by pass
    /// (`ir::fuel`); a lane that runs out stops and flags its pipeline. The same ceiling as a
    /// lane left as written: [`TargetLimits::lane_line_work`].
    pub fn lane_fuel(&self) -> u64 {
        self.lane_line_work()
    }

    /// The line's work at the idle clock: what one lane does in [`STEP_MS`] once the GPU has
    /// lowered its clock. A draw whose lanes' bound is past it is named in every build (W0016),
    /// the list of draws to bring within it.
    pub fn lane_idle_work(&self) -> u64 {
        self.lane_line_work() * 10 / u64::from(self.idle_slowdown_tenths.max(10))
    }
}

/// The longest one compute dispatch may hold the GPU, in milliseconds (AGENTS.md, rule 4: "at
/// most about 50 ms" a part). The host refuses, before it runs, a dispatch whose bound says it
/// could take longer.
pub const STEP_MS: u32 = 50;

/// The most work (thousands of units of language.md §18.11's count) one dispatch may do in
/// `ms` milliseconds at `femtoseconds` a unit: its kernel's bound times its invocations.
pub const fn step_work(ms: u64, femtoseconds: u32) -> u32 {
    let units = ms * 1_000_000_000_000 / femtoseconds as u64 / 1000;
    if units > u32::MAX as u64 { u32::MAX } else { units as u32 }
}

impl TargetLimits {
    /// An Apple M4's (8 GPU cores) as wgpu 30's Metal backend reports them (`wrela-host
    /// --adapter`, 2026-11-15).
    pub const METAL: TargetLimits = TargetLimits {
        device: Limits {
            max_texture_size: 16_384,
            // `maxStorageBufferBindingSize`, 4 GiB less 4, the lesser of it and `maxBufferSize`.
            max_buffer_size: 4_294_967_292,
            max_storage_buffers_per_stage: 29,
            max_uniform_buffer_binding_size: u32::MAX,
            max_workgroup_storage_size: 32_768,
            max_workgroup_invocations: 1024,
            max_workgroup_size: [1024, 1024, 1024],
            max_workgroups_per_dimension: 65_535,
            features: REQUIRED_FEATURES,
            subgroup_min_size: 0,
            subgroup_max_size: 0,
            step_work: step_work(STEP_MS as u64, METAL_FS),
            screen: 0,
            headroom: 1000,
            epoch: 0,
        },
        storage_buffers_per_stage: 29,
        sampled_textures_per_stage: 96,
        samplers_per_stage: 16,
        storage_textures_per_stage: 32,
        varyings: 31,
        color_attachment_bytes: 64,
        // A kernel that reaches the device's table has two fewer (`table::METAL_BLOCK`,
        // `METAL_TABLE`).
        buffer_slots_per_stage: Some(31),
        step_femtoseconds: METAL_FS,
        step_width: 8192,
        // The M4's fastest timings of `wrela-host`'s calibration kernel (8,192 invocations of a
        // chain of 8,192 sines), warmed: 1.14 to 1.21 ms (2026-11-19).
        calibration_ns: 1_150_000,
        // A pass of 1.24 ms took 5.33 ms after a second idle (2026-11-20).
        idle_slowdown_tenths: 43,
    };

    /// The floor of the Vulkan devices wrela runs on (Phase 5's minimum spec, backend.md item
    /// 35): what pipelines are compiled for, and less than which the host refuses an adapter.
    /// Each is Vulkan Roadmap 2022's (every Vulkan 1.3 desktop device: NVIDIA, AMD, Intel, the
    /// Steam Deck) or under it where content needs no more, and never more than the three
    /// vendors' devices report: until 2026-11-19 it was the Radeon 5500U's own numbers, which
    /// an NVIDIA GPU (48 KiB of workgroup memory, a 64 KiB uniform range) is short of.
    /// What a device has above the floor a program reads at run time (`std::gpu::limits()`).
    pub const VULKAN: TargetLimits = TargetLimits {
        device: Limits {
            // Roadmap 2022 is 8192; desktop devices 16384 or more.
            max_texture_size: 16_384,
            // `maxStorageBufferRange`: Intel's integrated GPUs report 1 GiB.
            max_buffer_size: 1_073_741_824,
            // Roadmap 2022.
            max_storage_buffers_per_stage: 30,
            // Roadmap 2022 (`maxUniformBufferRange`), NVIDIA's.
            max_uniform_buffer_binding_size: 65_536,
            // Apple's 32 KiB, under NVIDIA's 48 and AMD's and Intel's 64 (the core floor is 16).
            max_workgroup_storage_size: 32_768,
            // Every desktop vendor's and Metal's (Roadmap 2022 is 256, and 64 along z: NVIDIA's).
            max_workgroup_invocations: 1024,
            max_workgroup_size: [1024, 1024, 64],
            max_workgroups_per_dimension: 65_535,
            features: REQUIRED_FEATURES,
            subgroup_min_size: 0,
            subgroup_max_size: 0,
            step_work: step_work(STEP_MS as u64, VULKAN_FS),
            screen: 0,
            headroom: 1000,
            epoch: 0,
        },
        // Roadmap 2022's per-stage descriptors.
        storage_buffers_per_stage: 30,
        sampled_textures_per_stage: 200,
        samplers_per_stage: 64,
        storage_textures_per_stage: 16,
        varyings: 31,
        color_attachment_bytes: 128,
        buffer_slots_per_stage: None,
        step_femtoseconds: VULKAN_FS,
        step_width: 4096,
        // The Radeon 5500U's (RADV), warmed: 1.19 to 1.25 ms (2026-11-19).
        calibration_ns: 1_180_000,
        // A pass of 0.088 ms took 0.361 ms after any idle of 5 ms or more (2026-11-20).
        idle_slowdown_tenths: 41,
    };
}

/// [`TargetLimits::step_femtoseconds`] on the M4: sines in a chain at their bound, the slowest
/// arithmetic measured (2.51 fs a unit; a chain of multiply-adds 1.93, four in a vector 1.48),
/// with no load beside it, so at the clock the GPU lowers to under a light load (2026-11-15).
/// One invocation alone: 20 to 22 ns a unit, so about 8,192 run side by side.
const METAL_FS: u32 = 2500;
/// [`TargetLimits::step_femtoseconds`] on the Radeon: a dependent chain of multiply-adds at its
/// bound, the slowest arithmetic measured (2.25 fs a unit; four chains in a vector 1.25, sines
/// 1.41), rounded up (2026-11-15). One invocation alone: 5.1 to 7.6 ns a unit, so about 4,096
/// run side by side.
const VULKAN_FS: u32 = 2500;

/// The GPU limits a program can ask for (`wrela.limit(index)`, by index in this order) and the
/// hosts check commands against. The host opens its device with the adapter's own limits, and
/// refuses an adapter with less than its target's ([`TargetLimits`]), what pipelines are
/// compiled for. A value over `u32::MAX` is given as `u32::MAX`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// `maxTextureDimension2D`.
    pub max_texture_size: u32,
    /// The lesser of `maxBufferSize` and `maxStorageBufferBindingSize`: every buffer can be bound
    /// as storage.
    pub max_buffer_size: u32,
    /// `maxStorageBuffersPerShaderStage`.
    pub max_storage_buffers_per_stage: u32,
    /// `maxUniformBufferBindingSize`.
    pub max_uniform_buffer_binding_size: u32,
    /// `maxComputeWorkgroupStorageSize`.
    pub max_workgroup_storage_size: u32,
    /// `maxComputeInvocationsPerWorkgroup`.
    pub max_workgroup_invocations: u32,
    /// `maxComputeWorkgroupSizeX`, `Y` and `Z`.
    pub max_workgroup_size: [u32; 3],
    /// `maxComputeWorkgroupsPerDimension`.
    pub max_workgroups_per_dimension: u32,
    /// The device's optional features, a bit each in [`FEATURES`]' order ([`feature_bit`]).
    pub features: u32,
    /// The least and the most invocations a subgroup has (`subgroupMinSize`, `subgroupMaxSize`),
    /// or 0 without `subgroups`.
    pub subgroup_min_size: u32,
    pub subgroup_max_size: u32,
    /// The most work one compute dispatch may do, in thousands of units of the compiler's count
    /// (language.md §18.11): its kernel's bound times its invocations (never fewer than the
    /// GPU's [`TargetLimits::step_width`]). The host refuses a dispatch past it before it runs;
    /// a program that dispatches heavy work cuts it into parts this size.
    pub step_work: u32,
    /// How the host shows the screen ([`SCREEN_LIMIT`]): bit 0 HDR, bit 1 Display P3. Not the
    /// device's: what the host did of the last `ConfigureScreen` it could.
    pub screen: u32,
    /// The display's headroom now, in thousandths ([`HEADROOM_LIMIT`]): how many times SDR
    /// white the brightest value the screen shows is (1000 on an SDR display, and offscreen).
    pub headroom: u32,
    /// How many times the device has been lost and made again ([`EPOCH_LIMIT`]): after a loss
    /// every buffer and texture is there at its handle and size, zeroed. Not the device's: the
    /// host's count.
    pub epoch: u32,
}

impl Limits {
    /// What a host without a GPU reports: its machine's target's ([`Target::HERE`]).
    pub const HERE: Limits = Target::HERE.limits().device;

    /// Whether these limits are at least `need`'s, each of them (features aside): the name of
    /// the first that isn't, else `None`.
    pub fn short_of(&self, need: &Limits) -> Option<&'static str> {
        let pairs = [
            ("max_texture_size", self.max_texture_size, need.max_texture_size),
            ("max_buffer_size", self.max_buffer_size, need.max_buffer_size),
            (
                "max_storage_buffers_per_stage",
                self.max_storage_buffers_per_stage,
                need.max_storage_buffers_per_stage,
            ),
            (
                "max_uniform_buffer_binding_size",
                self.max_uniform_buffer_binding_size,
                need.max_uniform_buffer_binding_size,
            ),
            (
                "max_workgroup_storage_size",
                self.max_workgroup_storage_size,
                need.max_workgroup_storage_size,
            ),
            (
                "max_workgroup_invocations",
                self.max_workgroup_invocations,
                need.max_workgroup_invocations,
            ),
            ("max_workgroup_size.x", self.max_workgroup_size[0], need.max_workgroup_size[0]),
            ("max_workgroup_size.y", self.max_workgroup_size[1], need.max_workgroup_size[1]),
            ("max_workgroup_size.z", self.max_workgroup_size[2], need.max_workgroup_size[2]),
            (
                "max_workgroups_per_dimension",
                self.max_workgroups_per_dimension,
                need.max_workgroups_per_dimension,
            ),
        ];
        pairs.into_iter().find(|(_, have, need)| have < need).map(|(name, _, _)| name)
    }

    /// How many there are: `wrela.limit` gives 0 for an index past the last.
    pub const COUNT: u32 = 16;

    /// Whether the device has feature `name` (WebGPU's).
    pub fn has(&self, name: &str) -> bool {
        feature_bit(name).is_some_and(|b| self.features & b != 0)
    }

    /// The limit at `index`, as `wrela.limit` gives it. [`SCREEN_LIMIT`] and [`HEADROOM_LIMIT`]
    /// are no limits of the device's but how the host shows the screen and the display's
    /// headroom: offscreen, 0 (SDR, sRGB) and 1000.
    pub fn get(&self, index: u32) -> u32 {
        match index {
            0 => self.max_texture_size,
            1 => self.max_buffer_size,
            2 => self.max_storage_buffers_per_stage,
            3 => self.max_uniform_buffer_binding_size,
            4 => self.max_workgroup_storage_size,
            5 => self.max_workgroup_invocations,
            6..=8 => self.max_workgroup_size[index as usize - 6],
            9 => self.max_workgroups_per_dimension,
            10 => self.features,
            11 => self.subgroup_min_size,
            12 => self.subgroup_max_size,
            13 => self.step_work,
            SCREEN_LIMIT => self.screen,
            HEADROOM_LIMIT => self.headroom,
            EPOCH_LIMIT => self.epoch,
            _ => 0,
        }
    }
}

/// `wrela.limit`'s index of how the host shows the screen (`std::gpu::screen_is`): bit 0, HDR
/// (an `rgba16float` canvas, tone mapped `extended`); bit 1, Display P3. It changes when the host
/// has done what a `ConfigureScreen` asked. (It was 13, which `step_work` took: `screen_is`
/// read the step's work as the screen until toolchain 0.33.)
pub const SCREEN_LIMIT: u32 = 14;

/// `wrela.limit`'s index of how many times the device has been lost and made again
/// (`std::gpu::device_epoch`, [`Limits::epoch`]).
pub const EPOCH_LIMIT: u32 = 16;

/// `wrela.limit`'s index of the display's headroom (`std::gpu::headroom`), in thousandths.
pub const HEADROOM_LIMIT: u32 = 15;

/// The optional WebGPU features the hosts request when the adapter has them, and report to a
/// program (`std::gpu::features`), in the order `wrela.features` sets their bits: WebGPU's names.
/// None is required: a program checks before it records what needs one. A pipeline's manifest
/// entry names the ones its shader needs (`requires`).
pub const FEATURES: [&str; 12] = [
    "shader-f16",
    "subgroups",
    "texture-compression-bc",
    "texture-compression-etc2",
    "texture-compression-astc",
    "float32-filterable",
    "float32-blendable",
    "rg11b10ufloat-renderable",
    "bgra8unorm-storage",
    "timestamp-query",
    "chromium-experimental-multi-draw-indirect",
    "indirect-first-instance",
];

/// The bit of `wrela.features`' answer that says a device has feature `name` ([`FEATURES`]).
pub fn feature_bit(name: &str) -> Option<u32> {
    FEATURES.iter().position(|f| *f == name).map(|i| 1 << i)
}

/// The features every host requires of its device, as [`feature_bit`]s:
/// `indirect-first-instance` alone.
pub const REQUIRED_FEATURES: u32 = 1 << 11;

/// The checked-in files generated from this crate, each as (path, contents): std's ([`wrela`])
/// and the test vectors ([`vectors`]). `cargo run -p wrela-abi --bin generate` writes them.
pub fn generated_files() -> [(PathBuf, String); 2] {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    [
        (root.join(wrela::WRELA_PATH), wrela::wrela()),
        (root.join(vectors::VECTORS_PATH), vectors::vectors()),
    ]
}

#[cfg(test)]
mod tests {
    /// A module's pair section reads back; one without, a cut one and not a module give none.
    #[test]
    fn a_pair_section_reads_back() {
        let header = b"\0asm\x01\0\0\0".to_vec();
        let id = super::pair_id(Some(b"client"), None);
        let mut wasm = header.clone();
        // A custom section: id 0, its size, the name's length and bytes, the eight bytes.
        let name = super::PAIR_SECTION.as_bytes();
        wasm.push(0);
        wasm.push((1 + name.len() + 8) as u8);
        wasm.push(name.len() as u8);
        wasm.extend_from_slice(name);
        wasm.extend_from_slice(&id.to_le_bytes());
        assert_eq!(super::read_pair(&wasm), Some(id));
        assert_eq!(super::read_pair(&header), None);
        assert_eq!(super::read_pair(&wasm[..wasm.len() - 1]), None);
        assert_eq!(super::read_pair(b"not a module"), None);
        // The ids of the two roles' modules differ by which is which.
        assert_ne!(super::pair_id(Some(b"a"), None), super::pair_id(None, Some(b"a")));
    }

    /// The required features' bit is `indirect-first-instance`'s, in [`super::FEATURES`]' order.
    #[test]
    fn required_features_are_indirect_first_instance() {
        assert_eq!(super::feature_bit("indirect-first-instance"), Some(super::REQUIRED_FEATURES));
    }

    #[test]
    fn a_pair_section_rejects_malformed_modules() {
        let mut wasm = b"\0asm\x01\0\0\0".to_vec();
        let name = super::PAIR_SECTION.as_bytes();
        wasm.extend([0, (1 + name.len() + 8) as u8, name.len() as u8]);
        wasm.extend(name);
        wasm.extend(7u64.to_le_bytes());
        assert_eq!(super::read_pair(&wasm), Some(7));

        let mut section_after_pair = wasm.clone();
        section_after_pair.extend([0, 1, 0]);
        assert_eq!(super::read_pair(&section_after_pair), Some(7));

        let mut bad_version = wasm.clone();
        bad_version[4] = 2;
        assert_eq!(super::read_pair(&bad_version), None);

        let mut truncated_after_pair = wasm.clone();
        truncated_after_pair.extend([0, 10]);
        assert_eq!(super::read_pair(&truncated_after_pair), None);

        // A section's length is u32 LEB128, including a padded representation.
        let mut oversized_length = wasm[..8].to_vec();
        oversized_length.extend([0, wasm[9] | 0x80, 0x80, 0x80, 0x80, 0x10]);
        oversized_length.extend_from_slice(&wasm[10..]);
        assert_eq!(super::read_pair(&oversized_length), None);
    }

    /// The checked-in generated files are this crate's output (std and the tests read them).
    #[test]
    fn checked_in_copies_are_current() {
        for (path, text) in super::generated_files() {
            let actual = std::fs::read_to_string(&path).unwrap_or_default();
            assert!(
                actual == text,
                "{} is stale; run `cargo run -p wrela-abi --bin generate`",
                path.display()
            );
        }
    }
}
