//! `wrela-host`: runs a build headless.
//!
//! - `wrela-host <build-dir> [--frames N] [--fps F] [--size WxH] [--input script.json]
//!   [--png out.png [--every K [--png-from N]]] [--rgba out.rgba] [--timestamps [--serial]] [--cpu]
//!   [--log out.ticks] [--pipelines] [--capture F out.gputrace] [--workers N]`: frames on the GPU (in lockstep
//!   with the program's ticker, if it has one), then the state hash; the phases the program
//!   timed (`std::time::phase`), if any, on stderr. `--pipelines` reports each pipeline on
//!   stderr as it's built (`pipeline_report`); `--capture` writes frame F's GPU work as a Metal
//!   GPU trace for Xcode (`capture`, macOS only).
//! - `wrela-host <build-dir> --until TEXT [--input script.json] [--frames N] [--size WxH]
//!   --png out.png`: frames until the program prints TEXT: that frame is the PNG (a capture's
//!   still).
//! - `wrela-host --no-gpu <build-dir> --ticks N [--input script.json] [--log out.ticks]
//!   [--workers N]`: the program's ticks alone, with no GPU and no frames: each tick's state
//!   hash.
//! - `wrela-host --replay <log> [--no-gpu] <build-dir> [--workers N]`: a tick log's ticks,
//!   each with its records, checking each state hash; it fails at the first that differs.
//! - `wrela-host --compare-ticks <a.ticks> <b.ticks>`: two tick logs, two builds' runs of one
//!   input: "equal", or the first tick whose state hash differs (G-sim). Unlike `--replay` it
//!   doesn't compare the logs' WASM hashes.
//! - `wrela-host <build-dir> --window [--present fifo|fifo-relaxed|mailbox|immediate]
//!   [--frames N] [--size WxH] [--input FILE] [--arrive FILE] [--record FILE] [--log FILE]`:
//!   the frames shown in a window (`window`), until it's closed, on play's schedule (the ticker on
//!   its own clock: `paced`), each frame taking a person's keys, pointer, wheel and gamepads
//!   (`live`) and `--input`'s events at it; `--arrive`'s events come as a person's do, at times
//!   on the clock, and it prints how late the frames took them; `--log` writes the tick log,
//!   which `--replay` replays, and checks that each event the frames took reached the sim on the
//!   first tick after its frame handed it over (the parry-tick test); `--record` writes the
//!   records each tick took as a script, which `--no-gpu --ticks N --input` runs to the same
//!   ticks (without a ticker, the frames' events at their frames).
//!   `--hdr` with a headless run lets the screen go HDR when the program asks
//!   (`std::gpu::screen`), as a test's `hdr=1` does; a window goes HDR where its display can.
//! - `wrela-host --adapter`: the GPU as `wrela_gpu` opens it: its name, driver, features and
//!   limits.
//! - `wrela-host --version`: its release and source revision.
//!
//! `--relaxed`, with any of them, runs a relaxed-SIMD build's relaxed module (`wrela build
//! --relaxed-simd`), as a browser that runs relaxed SIMD would.
//!
//! A build of this binary without the `gpu` feature runs ticks only, with `--no-gpu` or
//! `--replay` (an x86-64 build run under Rosetta, say, which checks the sim's bits on x86's code
//! generation); `--compare-ticks` and `--version` work in either build.

use std::path::PathBuf;
use std::process::ExitCode;
#[cfg(feature = "gpu")]
use wrela_host::PresentMode;
use wrela_host::{CpuBuild, TickLog};

/// Without the GPU there's no window to present to: `--window` fails when it runs.
#[cfg(not(feature = "gpu"))]
#[derive(Clone, Copy)]
enum PresentMode {
    Fifo,
}

#[cfg(not(feature = "gpu"))]
impl PresentMode {
    fn parse(_: &str) -> Option<PresentMode> {
        Some(PresentMode::Fifo)
    }
}

const USAGE: &str = "usage:
  wrela-host <build-dir> [--frames N] [--fps F] [--size WxH] [--input script.json] [--png out.png [--every K [--png-from N]]] [--rgba out.rgba] [--timestamps [--serial]] [--keep-clocked] [--cpu] [--targets DIR] [--log out.ticks] [--pipelines] [--steps] [--capture F out.gputrace] [--workers N]
  wrela-host <build-dir> --until TEXT [--input script.json] [--frames N] [--fps F] [--size WxH] --png out.png [--pipelines] [--workers N]
  wrela-host --no-gpu <build-dir> --ticks N [--input script.json] [--log out.ticks] [--workers N]
  wrela-host --replay <log> [--no-gpu] <build-dir> [--workers N]
  wrela-host --compare-ticks <a.ticks> <b.ticks>
  wrela-host <build-dir> --window [--present fifo|fifo-relaxed|mailbox|immediate] [--frames N] [--fps F] [--size WxH] [--input script.json] [--arrive script.json] [--record out.json] [--log out.ticks] [--workers N]
  wrela-host --adapter

With a GPU: runs the build's frame(i / fps, width, height) for i in 0..N (default 1 frame at 60
fps, 640x360), its ticker (if it starts one) in lockstep, then prints the state hash: FNV-1a 64
over every submitted byte, 16 hex digits. --input gives the program a script of input events
(runtime/abi `input`). --png and --rgba write the last frame (raw RGBA8, rows top to bottom);
with --every K, --png writes every Kth frame too, numbered (out-0000.png, out-0001.png, ...): a
clip's frames. --png-from N omits numbered captures before frame N, while still running those
frames; the capture schedule and numbering stay the same.
--until runs frames until the program prints a line holding TEXT, and the frame that prints it
is the PNG; it fails if none does within --frames.
--timestamps prints each dispatch's and pass's GPU time to stderr, and each frame's span (its
first start to its last end); --serial runs each pass and dispatch alone, so each time is its own.
--keep-clocked keeps the GPU's clock up while the program works on the CPU: a thread with a
device of its own runs a piece of ~0.1 ms whenever the GPU has had nothing in flight for 1 ms,
so a pass after the program's CPU work runs at the GPU's height, not many times slower (a judge's
or a timing run's; it costs power, and a game is not run so).
--cpu prints each frame's CPU time on the host's thread to stderr (`frame N  cpu total T ms
program P check C record R blocked B`): the program's WASM and ticks, checking what it
submits, recording the GPU's work, and waiting for the GPU. Without --timestamps: with them,
the host waits for each submission's timestamps, and that wait is blocked time.
--steps prints a line to stderr for each dispatch and draw as it's recorded: its frame, label
and pipeline, a dispatch's invocations and its kernel's most work an invocation (work.json), a
draw's vertices and instances.
--targets DIR writes, after the last frame, each labelled pass's target as its own texels and as a
PNG, and every buffer's bytes, with targets.json saying which is which (`Host::dump_targets`):
what tools/hosts compares two hosts by.
--pipelines prints a line to stderr for each pipeline as it's built: its MSL file
(pipeline_N.metal) and entry points; a compute pipeline's declared workgroup size and, on Metal,
its SIMD width, the most threads a threadgroup of it may have and its threadgroup memory; a
render pipeline's entry points and target variants. The most threads comes as built (capped at
the workgroup's size, which the shader declares to Metal) and uncapped (as many as its
registers allow: below the device's, they limit how many threads a core holds; an Apple M4,
which allots registers as a kernel runs, reports the device's for any kernel). Native
diagnostics only: a browser reports none of this. On Metal each compute pipeline is compiled
twice more to ask, so loading takes longer.
--capture F out.gputrace captures frame F's GPU work (0-based, below --frames) to a Metal GPU
trace, which Xcode opens to replay the frame with its performance counters: each pipeline's
registers, occupancy and limiters. macOS only (Metal's capture manager). Metal captures to a
file only in a process started with MTL_CAPTURE_ENABLED=1, which wrela-host sets for itself
with --capture; the file mustn't exist yet.

--no-gpu runs the ticker's ticks alone, with no GPU and no frames, and prints the first world's
state hash and each tick's (`tick K HASH`). --log writes the ticks as a tick log (runtime/abi
`ticks`). --replay runs a tick log's ticks with their records, checks each state hash, and fails
at the first that differs, naming it. --compare-ticks reads two tick logs (two builds run on the
same input) and prints `equal`, or the first tick whose state hash differs; it exits 1 if they
differ. --workers sets the threads that run parallel work (the
program's own among them). --relaxed runs a relaxed-SIMD build's relaxed module.";

struct Args {
    dir: PathBuf,
    /// `--frames`: headless, 1 if not given ([`Args::frames`]); in a window, until it's closed.
    frames: Option<u32>,
    fps: f64,
    size: (u32, u32),
    png: Option<PathBuf>,
    every: Option<u32>,
    png_from: Option<u32>,
    rgba: Option<PathBuf>,
    targets: Option<PathBuf>,
    input: Option<PathBuf>,
    /// `--record FILE`: in a window, what the frames took, written as a script.
    record: Option<PathBuf>,
    /// `--arrive FILE`: in a window, a script's events arriving as a person's do.
    arrive: Option<PathBuf>,
    until: Option<String>,
    timestamps: bool,
    serial: bool,
    cpu: bool,
    pipelines: bool,
    capture: Option<(u32, PathBuf)>,
    no_gpu: bool,
    ticks: Option<u32>,
    log: Option<PathBuf>,
    replay: Option<PathBuf>,
    workers: u32,
    relaxed: bool,
    /// `--window`: shown in a window, presenting as this says.
    window: Option<PresentMode>,
    /// `--hdr`: the screen goes HDR offscreen when the program asks (as a test's `hdr=1`).
    hdr: bool,
}

fn parse(mut argv: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut dir = None;
    let mut args = Args {
        dir: PathBuf::new(),
        frames: None,
        fps: 60.0,
        size: (640, 360),
        png: None,
        every: None,
        png_from: None,
        rgba: None,
        targets: None,
        input: None,
        record: None,
        arrive: None,
        until: None,
        timestamps: false,
        serial: false,
        cpu: false,
        pipelines: false,
        capture: None,
        no_gpu: false,
        ticks: None,
        log: None,
        replay: None,
        workers: 0,
        relaxed: false,
        window: None,
        hdr: false,
    };
    while let Some(arg) = argv.next() {
        let mut value = |flag: &str| argv.next().ok_or_else(|| format!("{flag} needs a value"));
        let whole = |flag: &str, v: String| -> Result<u32, String> {
            v.parse().map_err(|_| format!("{flag} needs a whole number"))
        };
        match arg.as_str() {
            "--frames" => args.frames = Some(whole("--frames", value("--frames")?)?),
            "--hdr" => args.hdr = true,
            "--window" => {
                args.window.get_or_insert(PresentMode::Fifo);
            }
            "--present" => {
                let mode = value("--present")?;
                args.window = Some(PresentMode::parse(&mode).ok_or_else(|| {
                    format!("--present is fifo, fifo-relaxed, mailbox or immediate, not {mode}")
                })?);
            }
            "--ticks" => args.ticks = Some(whole("--ticks", value("--ticks")?)?),
            "--workers" => args.workers = whole("--workers", value("--workers")?)?,
            "--fps" => {
                args.fps =
                    value("--fps")?.parse().map_err(|_| "--fps needs a number".to_string())?;
                if !(args.fps.is_finite() && args.fps > 0.0) {
                    return Err("--fps must be positive".into());
                }
            }
            "--size" => {
                let v = value("--size")?;
                let parsed =
                    v.split_once('x').and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)));
                args.size =
                    parsed.ok_or_else(|| format!("--size needs WIDTHxHEIGHT, not {v:?}"))?;
            }
            "--png" => args.png = Some(value("--png")?.into()),
            "--every" => args.every = Some(whole("--every", value("--every")?)?.max(1)),
            "--png-from" => args.png_from = Some(whole("--png-from", value("--png-from")?)?),
            "--rgba" => args.rgba = Some(value("--rgba")?.into()),
            "--targets" => args.targets = Some(value("--targets")?.into()),
            "--input" => args.input = Some(value("--input")?.into()),
            "--record" => args.record = Some(value("--record")?.into()),
            "--arrive" => args.arrive = Some(value("--arrive")?.into()),
            "--until" => args.until = Some(value("--until")?),
            "--log" => args.log = Some(value("--log")?.into()),
            "--replay" => args.replay = Some(value("--replay")?.into()),
            "--timestamps" => args.timestamps = true,
            "--serial" => args.serial = true,
            #[cfg(feature = "gpu")]
            "--keep-clocked" => wrela_host::keep_clocked(),
            "--cpu" => args.cpu = true,
            "--pipelines" => args.pipelines = true,
            #[cfg(feature = "gpu")]
            "--steps" => wrela_host::step::log_steps(),
            "--capture" => {
                let frame = whole("--capture", value("--capture")?)?;
                args.capture = Some((frame, value("--capture")?.into()));
            }
            "--no-gpu" => args.no_gpu = true,
            "--relaxed" => args.relaxed = true,
            "-h" | "--help" => return Err(USAGE.into()),
            flag if flag.starts_with("--") => {
                return Err(format!("unknown option {flag}\n\n{USAGE}"));
            }
            _ if dir.is_none() => dir = Some(PathBuf::from(arg)),
            _ => return Err(format!("unexpected argument {arg:?}\n\n{USAGE}")),
        }
    }
    args.dir = dir.ok_or_else(|| USAGE.to_string())?;
    if args.until.is_some() && (args.png.is_none() || args.every.is_some() || args.log.is_some()) {
        return Err(format!("--until needs --png, and takes no --every or --log\n\n{USAGE}"));
    }
    if args.every.is_some() && args.png.is_none() {
        return Err(format!("--every needs --png\n\n{USAGE}"));
    }
    if args.png_from.is_some() && args.every.is_none() {
        return Err(format!("--png-from needs --png and --every\n\n{USAGE}"));
    }
    if args.no_gpu && args.replay.is_none() && args.ticks.is_none() {
        return Err(format!("--no-gpu needs --ticks N\n\n{USAGE}"));
    }
    if let Some((frame, path)) = &args.capture {
        if args.until.is_some() || args.no_gpu || args.replay.is_some() {
            return Err(format!(
                "--capture runs with frames alone, not --until, --no-gpu or --replay\n\n{USAGE}"
            ));
        }
        if *frame >= args.frames() {
            return Err(format!(
                "--capture {frame} is past the last frame: --frames is {}",
                args.frames()
            ));
        }
        // Before the frames run, not when the capture starts (Metal checks again then).
        if path.extension().is_none_or(|e| e != "gputrace") || path.exists() {
            return Err(format!(
                "--capture writes a new .gputrace: {} isn't one, or exists",
                path.display()
            ));
        }
    }
    Ok(args)
}

impl Args {
    /// The frames a headless run runs.
    fn frames(&self) -> u32 {
        self.frames.unwrap_or(1)
    }
}

/// The host's options, as the flags say.
#[cfg(feature = "gpu")]
fn options(args: &Args) -> wrela_host::Options {
    use wrela_host::Timing;
    let timing = match (args.timestamps, args.serial) {
        (true, true) => Timing::Serial,
        (true, false) => Timing::Span,
        // --serial says how --timestamps times: alone, it asks for nothing.
        (false, _) => Timing::Off,
    };
    wrela_host::Options {
        timing,
        report_pipelines: args.pipelines,
        capture: args.capture.clone(),
        workers: args.workers,
        defer_init: true,
        relaxed_simd: args.relaxed,
        screen_hdr: args.hdr,
        ..wrela_host::Options::default()
    }
}

/// What went wrong: the host's error, or a replay that didn't match.
enum Failure {
    Host(wrela_host::Error),
    Replay(wrela_host::ReplayError),
}

impl From<wrela_host::Error> for Failure {
    fn from(e: wrela_host::Error) -> Failure {
        Failure::Host(e)
    }
}

fn read_script(path: Option<&PathBuf>) -> Result<Vec<wrela_host::Scripted>, wrela_host::Error> {
    match path {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| wrela_host::Error::io(path, e))?;
            wrela_host::parse_script(&text)
                .map_err(|why| wrela_host::Error::Program(format!("{}: {why}", path.display())))
        }
        None => Ok(Vec::new()),
    }
}

fn write_log(path: &PathBuf, log: &TickLog) -> Result<(), wrela_host::Error> {
    std::fs::write(path, log.encode()).map_err(|e| wrela_host::Error::io(path, e))
}

/// `--compare-ticks`: two tick logs' first difference, or `equal` (exit 0).
fn compare_ticks(a: &str, b: &str) -> ExitCode {
    let read = |path: &str| -> Result<TickLog, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        TickLog::decode(&bytes).map_err(|e| format!("{path}: {e}"))
    };
    match (read(a), read(b)) {
        (Ok(x), Ok(y)) => {
            let c = wrela_abi::ticks::TickComparison::of(&x, &y);
            println!("{c}");
            if matches!(c, wrela_abi::ticks::TickComparison::Equal { .. }) {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

/// `--replay`: a tick log's ticks, each hash checked.
fn replay(args: &Args, path: &PathBuf) -> Result<(), Failure> {
    let bytes = std::fs::read(path).map_err(|e| wrela_host::Error::io(path, e))?;
    let log = TickLog::decode(&bytes)
        .map_err(|e| wrela_host::Error::Program(format!("{}: {e}", path.display())))?;
    let built = CpuBuild::load_choosing(&args.dir, args.relaxed)?;
    let n = built.replay(&log, args.workers).map_err(Failure::Replay)?;
    println!("{n} ticks replayed: every state hash is the log's");
    Ok(())
}

/// `--no-gpu`: the ticks alone.
fn ticks_alone(args: &Args) -> Result<(), Failure> {
    let script = read_script(args.input.as_ref())?;
    let built = CpuBuild::load_choosing(&args.dir, args.relaxed)?;
    let log = built.record_ticks(args.ticks.unwrap_or(0), &script, args.workers)?;
    println!("first {}", wrela_abi::hash::hex(log.first));
    for (k, t) in log.ticks.iter().enumerate() {
        println!("tick {k} {}", wrela_abi::hash::hex(t.hash));
    }
    if let Some(path) = &args.log {
        write_log(path, &log)?;
    }
    Ok(())
}

/// `--until`: frames until the program prints the text, that frame the PNG.
#[cfg(feature = "gpu")]
fn still(args: &Args, until: &str) -> Result<(), Failure> {
    use wrela_host::{Host, Options};
    let script = read_script(args.input.as_ref())?;
    let options = Options {
        workers: args.workers,
        relaxed_simd: args.relaxed,
        report_pipelines: args.pipelines,
        ..Options::default()
    };
    let mut host = Host::load_with(&args.dir, &options)?;
    let (w, h) = args.size;
    let run = host.run_lockstep_until(args.frames(), args.fps, w, h, &script, until)?;
    drop(host); // release the GPU (and its lock) before writing files
    let Some(run) = run else {
        return Err(Failure::Host(wrela_host::Error::Program(format!(
            "it didn't print {until:?} within {} frames",
            args.frames()
        ))));
    };
    if let Some(path) = &args.png {
        run.write_png(path)?;
    }
    println!("{}", run.hash_hex());
    Ok(())
}

#[cfg(feature = "gpu")]
fn frames(args: &Args) -> Result<(), Failure> {
    use wrela_host::Host;
    if let Some(until) = &args.until {
        return still(args, until);
    }
    let script = read_script(args.input.as_ref())?;
    let loading = std::time::Instant::now();
    let mut host = Host::load_with(&args.dir, &options(args))?;
    let loaded_ms = loading.elapsed().as_secs_f64() * 1000.0;
    let logging = args.log.is_some();
    host.want_hashes(logging);
    host.init()?;
    let (w, h) = args.size;
    // With --every, each Kth frame as it's drawn: a clip's frames.
    let stem = args.png.as_ref().map(|p| p.with_extension("")).unwrap_or_default();
    let every = args.every.unwrap_or(0);
    let (run, log) = host.run_lockstep_seen_from(
        args.frames(),
        args.fps,
        w,
        h,
        &script,
        logging,
        every,
        args.png_from.unwrap_or(0),
        |i, rgba| {
            let path = PathBuf::from(format!("{}-{:04}.png", stem.display(), i / every));
            wrela_host::image::write_png(&path, w, h, rgba)
        },
    )?;
    if let Some(dir) = &args.targets {
        host.dump_targets(dir)?;
    }
    let (pipelines_built, pipelines_ms) = (host.pipelines_built(), host.pipelines_build_ms());
    // The phases the program timed (`std::time::phase`), over every frame and tick.
    let traced = host.trace().take();
    drop(host); // release the GPU (and its lock) before writing files
    if !traced.phases.is_empty() {
        eprintln!("phases: {}", traced.phases_line());
    }
    if let Some(path) = &args.png {
        run.write_png(path)?;
    }
    if let Some(path) = &args.rgba {
        std::fs::write(path, &run.frame).map_err(|e| wrela_host::Error::io(path, e))?;
    }
    if let (Some(path), Some(log)) = (&args.log, &log) {
        write_log(path, log)?;
    }
    for t in &run.timings {
        eprintln!("frame {:>4}  {:<24} {:>10.3} us", t.frame, t.label, t.nanos / 1000.0);
    }
    for (frame, ms) in wrela_host::frame_spans(&run.timings) {
        eprintln!("frame {frame:>4}  span {ms:.3} ms");
    }
    if args.cpu {
        eprintln!(
            "load  {loaded_ms:.1} ms, of which pipelines {} built in {pipelines_ms:.1} ms",
            pipelines_built
        );
        for c in &run.cpu {
            eprintln!(
                "frame {:>4}  cpu total {:.3} ms program {:.3} check {:.3} record {:.3} blocked {:.3}",
                c.frame, c.total, c.program, c.check, c.record, c.blocked
            );
        }
    }
    println!("{}", run.hash_hex());
    Ok(())
}

/// `--adapter`: the GPU as `wrela_gpu` opens it, its features and its limits.
#[cfg(feature = "gpu")]
fn adapter() -> ExitCode {
    match wrela_host::adapter_report() {
        Ok(report) => {
            print!("{report}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn adapter() -> ExitCode {
    eprintln!("error: this wrela-host was built without the `gpu` feature");
    ExitCode::FAILURE
}

#[cfg(not(feature = "gpu"))]
fn frames(_: &Args) -> Result<(), Failure> {
    Err(Failure::Host(wrela_host::Error::Gpu(
        "this wrela-host was built without the `gpu` feature: run ticks alone with --no-gpu, or --replay".into(),
    )))
}

fn run(args: &Args) -> Result<(), Failure> {
    if let Some(path) = &args.replay {
        return replay(args, path);
    }
    if args.no_gpu {
        return ticks_alone(args);
    }
    if let Some(present) = args.window {
        return windowed(args, present);
    }
    frames(args)
}

/// `--window`: the frames shown in a window until it's closed (or `--frames` given, ran).
#[cfg(feature = "gpu")]
fn windowed(args: &Args, present: PresentMode) -> Result<(), Failure> {
    use wrela_host::{Host, Options, Timing, window};
    // A window's frames aren't timed or captured, and its screen goes HDR where its display can.
    let options =
        Options { timing: Timing::Off, capture: None, screen_hdr: false, ..options(args) };
    let mut host = Host::load_with(&args.dir, &options)?;
    let logging = args.log.is_some();
    host.want_hashes(logging || args.record.is_some());
    host.init()?;
    // A tick log is the ticker's, as a headless run's is: without one, `--log` fails rather than
    // writing nothing.
    if logging && host.ticker_hz().is_none() {
        let e = "--log writes the ticker's ticks, and it starts no ticker in `init` (`std::tick::start`)";
        return Err(Failure::Host(wrela_host::Error::Program(e.into())));
    }
    let script = read_script(args.input.as_ref())?;
    let arrive = read_script(args.arrive.as_ref())?;
    if arrive.iter().any(|s| !matches!(s.at, wrela_host::At::Frame(_))) {
        let e = "--arrive's events arrive at frames' times: a tick's record can't arrive";
        return Err(Failure::Host(wrela_host::Error::Program(e.into())));
    }
    let checked = logging && script.iter().all(|s| matches!(s.at, wrela_host::At::Frame(_)));
    // `--record` writes the ticks' records from the log, if the program has a ticker.
    let logging = logging || args.record.is_some();
    let run = window::WindowRun {
        frames: args.frames,
        fps: args.fps,
        size: args.size,
        present,
        script,
        arrive,
        keep: args.record.is_some() || checked,
        log: logging && host.ticker_hz().is_some(),
    };
    let ran = window::run(&mut host, &run)?;
    if let (
        Some((median, p90, worst)),
        Some((s50, s90, sworst)),
        Some((before, _, _)),
        Some(first),
        Some(rate),
    ) = (ran.latency(), ran.settled(), ran.before_present(), ran.first(), ran.rate())
    {
        println!(
            "latency, input sampled to shown: median {median:.2} ms (to presented {before:.2} ms), 90th percentile {p90:.2} ms, worst {worst:.2} ms (the {}th shown) over {} frames after the first ({first:.0} ms)",
            ran.latencies.iter().position(|&l| l == worst).unwrap_or(0),
            ran.latencies.len() - 1
        );
        println!(
            "settled (the second half): median {s50:.2} ms, 90th percentile {s90:.2} ms, worst {sworst:.2} ms; {rate:.1} frames a second; refresh {:.2} ms; {} missed their refresh",
            ran.period, ran.misses
        );
        println!("pacer at the end: {}", ran.pacer);
    }
    if ran.longest_hold > 0.0 {
        println!(
            "held asleep by the system past its wakes by more than a refresh: {:.0} ms in all, {:.0} ms at the longest (a window hidden or its display asleep: the frames went as it woke)",
            ran.held * 1e3,
            ran.longest_hold * 1e3
        );
    }
    if let Some(a) = ran.arrivals() {
        println!(
            "arrived as a person's: {} events, taken {:.2} ms after (median), {:.2} ms (worst); {} taken by a later frame than the first that sampled after them{}",
            a.count,
            a.median,
            a.worst,
            a.late,
            if a.late > 0 { format!(" (by up to {:.2} ms)", a.late_by) } else { String::new() }
        );
    }
    if let Some(path) = &args.record {
        // With a ticker, the records each tick took, at their ticks: `--no-gpu --ticks N
        // --input` runs the same ticks. Without, the frames' events at their frames.
        let recorded = match &ran.log {
            Some(log) => log.script(),
            None => ran.taken.iter().map(|(s, _)| s.clone()).collect(),
        };
        let text = wrela_host::write_script(&recorded).map_err(wrela_host::Error::Program)?;
        std::fs::write(path, text).map_err(|e| wrela_host::Error::io(path, e))?;
        println!("{} events written to {}", recorded.len(), path.display());
    }
    if let (Some(path), Some(log)) = (&args.log, &ran.log) {
        write_log(path, log)?;
        if ran.dropped > 0 {
            println!("ticks: {} dropped, behind the clock", ran.dropped);
        }
        if checked {
            match window::check_ticks(log, &ran.took, &ran.taken) {
                Ok(r) => println!(
                    "ticks: {} events the frames took each reached the sim on the first tick after their frame handed them over, {:.2} ms after (median), {:.2} ms (worst); {} ticks",
                    r.count,
                    r.median,
                    r.worst,
                    log.ticks.len()
                ),
                Err(why) => {
                    let e = format!("an event didn't reach the sim on its tick: {why}");
                    return Err(Failure::Host(wrela_host::Error::Program(e)));
                }
            }
        } else {
            println!("ticks: not checked (the script has ticks' records)");
        }
    }
    if let Some(v) = &ran.voice {
        println!(
            "voice: on {} ({} channels, 48 kHz): {} quanta in {} callbacks of {} to {} frames; the slowest took {:.1}% of the time its buffer plays, {} took longer; {} found the voice busy and played silence{}",
            v.device,
            v.channels,
            v.quanta,
            v.callbacks,
            v.frames.0,
            v.frames.1,
            v.worst * 100.0,
            v.over,
            v.busy,
            if v.errors.is_empty() {
                String::new()
            } else {
                format!("; the device said: {}", v.errors.join("; "))
            }
        );
    }
    println!("{} frames shown; state hash {:016x}", ran.frames, host.hash());
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn windowed(_: &Args, _: PresentMode) -> Result<(), Failure> {
    Err(Failure::Host(wrela_host::Error::Gpu(
        "this wrela-host was built without the `gpu` feature".into(),
    )))
}

/// Lets Metal capture to a file (`--capture`): `MTL_CAPTURE_ENABLED=1`, which Metal reads when
/// the process first opens a device, if the environment doesn't already say.
#[cfg(feature = "gpu")]
#[allow(unsafe_code)]
fn enable_capture() {
    if std::env::var_os(wrela_host::capture::ENABLE).is_none() {
        // SAFETY: called from `main` before any thread starts or any device opens, so nothing
        // reads the environment while it changes.
        unsafe { std::env::set_var(wrela_host::capture::ENABLE, "1") };
    }
}

#[cfg(not(feature = "gpu"))]
fn enable_capture() {}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if let [flag, a, b] = argv.as_slice()
        && flag == "--compare-ticks"
    {
        return compare_ticks(a, b);
    }
    if argv == ["--adapter"] {
        return adapter();
    }
    if argv == ["--version"] {
        let revision = option_env!("WRELA_REVISION").unwrap_or("development build");
        println!("wrela-host {} ({revision})", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    let args = match parse(argv.into_iter()) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    if args.capture.is_some() {
        enable_capture();
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Host(e)) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
        Err(Failure::Replay(e)) => {
            eprintln!("the replay failed: {e}");
            ExitCode::FAILURE
        }
    }
}
