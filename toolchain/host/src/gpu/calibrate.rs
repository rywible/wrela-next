//! Device unit calibration and the optional `--keep-clocked` GPU warmer.

use super::*;
use std::sync::atomic::Ordering;

/// The calibration kernel's length: each invocation's chain of sines (`Gpu::calibration_ns`).
pub(super) const CALIBRATION_CHAIN: u32 = 8192;
/// Its workgroups of 64: 8,192 invocations, at least any GPU's width here.
pub(super) const CALIBRATION_GROUPS: u32 = 128;
/// How long the calibration kernel runs before it's timed (ms).
const CALIBRATION_WARM_MS: f64 = 500.0;
/// While the host times its calibration kernel (its own device: no calibration of that one).
pub(super) static CALIBRATING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether each device the host opens is kept clocked while the program works on the CPU
/// (`--keep-clocked`; [`Warmer`]).
pub(super) static KEEP_CLOCKED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Keeps the GPU clocked while the program works on the CPU, for every device opened after it
/// (`wrela-host --keep-clocked`): see [`Warmer`].
pub fn keep_clocked() {
    KEEP_CLOCKED.store(true, Ordering::SeqCst);
}

thread_local! {
    /// Whether this thread is a warmer's (its own device is not kept clocked).
    pub(super) static WARMING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The warmer's kernel: the calibration kernel's chain, shorter (a piece of about 0.04 ms on
/// the M4 at its height). Measured on the M4 (`how_much_a_pass_slows_after_idle`): pieces of
/// 128 to 512 a chain every 0.1 ms kept the clock up; 1024 every 0.25 ms or 1 ms, 8192 every
/// 1 ms, and 256 every 0.3 ms did not (the clock falls in gaps of a few tenths of a ms).
const WARM_CHAIN: u32 = 256;
/// How long the watched device is idle before the warmer begins (ms).
const WARM_IDLE_MS: f64 = 1.0;
/// The warmer's gap between its pieces, and how often it looks whether the device is idle (ms).
const WARM_GAP_MS: f64 = 0.1;

/// A GPU left idle lowers its clock, and a pass after is slow for real, by its own stamps (the
/// M4: a pass of 1.2 ms ran 4.3 times as long after a second idle, and up to 7 times in a run of
/// passes each after 20 ms of the program's CPU work; one of 12 ms, 1.6 times at the median and
/// 2.7 at most in that run): the guard, which holds a pass to what it held the GPU, then sees a
/// pass run many times its cost. A warmer keeps the clock up from outside the program: a thread
/// with a device of its own that, whenever the watched device has had nothing in flight for
/// `WARM_IDLE_MS`, runs a piece of the calibration kernel (`WARM_CHAIN`, ~0.04 ms), waits for
/// it, and looks again after `WARM_GAP_MS`. Each piece is far under the line, and none begins
/// while the watched device has work in flight, so a pass waits at most one piece. Off by
/// default: it costs a tenth of a CPU core and the GPU's power at its height through every gap,
/// and a game paces its frames and is not judged by its first pass after a gap.
pub(super) struct Warmer {
    stop: Arc<std::sync::atomic::AtomicBool>,
    pieces: Arc<std::sync::atomic::AtomicU64>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Warmer {
    pub(super) fn start(watched: wrela_gpu::Device) -> Warmer {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pieces = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let (s, p) = (stop.clone(), pieces.clone());
        let thread = std::thread::Builder::new()
            .name("keep-clocked".into())
            .spawn(move || {
                WARMING.with(|w| w.set(true));
                if let Err(e) = Warmer::warm(&watched, &s, &p) {
                    eprintln!("wrela-host: keeping the GPU clocked stopped: {e}");
                }
            })
            .ok();
        Warmer { stop, pieces, thread }
    }

    fn warm(
        watched: &wrela_gpu::Device,
        stop: &std::sync::atomic::AtomicBool,
        pieces: &std::sync::atomic::AtomicU64,
    ) -> Result<()> {
        let (manifest, shaders) = calibration_kernel(WARM_CHAIN)?;
        let mut gpu = Gpu::new(&manifest, &shaders, Timing::Off, false)?;
        let n = CALIBRATION_GROUPS * 64;
        gpu.create_buffer(1, 4 * n);
        let gap = std::time::Duration::from_secs_f64(WARM_GAP_MS / 1e3);
        let mut busy = std::time::Instant::now();
        while !stop.load(Ordering::Acquire) {
            if watched.completed() < watched.submitted() {
                busy = std::time::Instant::now();
            } else if busy.elapsed().as_secs_f64() * 1e3 >= WARM_IDLE_MS {
                gpu.dispatch(
                    0,
                    Ok([CALIBRATION_GROUPS, 1, 1]),
                    &[Binding::range(1, 0, 4 * n)],
                    &[],
                )?;
                gpu.flush()?;
                wait(&gpu.device)?;
                pieces.fetch_add(1, Ordering::Relaxed);
            }
            std::thread::sleep(gap);
        }
        gpu.check_errors()
    }
}

impl Drop for Warmer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        eprintln!(
            "wrela-host: kept the GPU clocked: {} pieces of the warmer's kernel",
            self.pieces.load(Ordering::Relaxed)
        );
    }
}

/// The calibration kernel: `x = sin(x) * 0.999 + 0.001`, `chain` times a chain,
/// from the invocation's index, its result written so nothing is left out. Its bound is stated
/// (a sine, a multiply, an add and the loop's count, 32 a pass, as the compiler counts them),
/// so the guard weighs it as any kernel.
pub(super) fn calibration_kernel(chain: u32) -> Result<(Manifest, Vec<Vec<u8>>)> {
    use wrela_abi::manifest::{BindingStage, Pipeline};
    use wrela_shader::build::{BinOp, Builder, Builtin, BuiltinInput, Scalar};
    let mut manifest = Manifest::new("calibrate.wasm");
    manifest.pipelines.push(Pipeline {
        name: "calibrate".into(),
        shader: String::new(),
        stage: Stage::Compute { entry: "calibrate".into(), workgroup_size: [64, 1, 1] },
        uniform: None,
        bindings: vec![ResourceBinding {
            binding: 0,
            kind: BindingKind::ReadWrite,
            stage: BindingStage::Both,
            format: None,
        }],
        debug_flag: None,
        requires: Vec::new(),
        conditional: false,
        standin: None,
        sizes: None,
        kept: 0,
        work: vec![Some(u64::from(chain) * 32 + 64)],
        unbounded: Vec::new(),
    });
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let f = b.scalar(Scalar::F32);
    let u3 = b.vector(Scalar::U32, 3);
    let out = b.storage("out", 0, f, true);
    b.compute("calibrate", [64, 1, 1], &[BuiltinInput::GlobalInvocationId], |k| {
        let id = k.input(0, u3);
        let i = k.extract(u, id, 0);
        let x = k.local("x", f);
        let n = k.local("n", u);
        let start = k.convert(f, i, Scalar::F32);
        k.store(x.clone(), start);
        let zero = k.u32(0);
        k.store(n.clone(), zero);
        k.loop_(|k| {
            let c = k.load(u, n.clone());
            let limit = k.u32(chain);
            let done = k.binary(BinOp::Ge, c, limit);
            k.if_(done, |k| k.break_(), |_| {});
            let v = k.load(f, x.clone());
            let s = k.builtin(f, Builtin::Sin, &[v]);
            let a = k.f32(0.999);
            let m = k.binary(BinOp::Mul, s, a);
            let b2 = k.f32(0.001);
            let y = k.binary(BinOp::Add, m, b2);
            k.store(x.clone(), y);
            let one = k.u32(1);
            let c2 = k.binary(BinOp::Add, c, one);
            k.store(n.clone(), c2);
        });
        let r = k.load(f, x);
        k.store(Builder::element(out, i), r);
        k.ret(None);
    });
    let target = wrela_abi::Target::HERE;
    let p = &mut manifest.pipelines[0];
    p.shader = format!("calibrate.{}", wrela_shader::extension(target));
    let shader =
        b.write(p, target).map_err(|e| Error::Gpu(format!("the calibration kernel: {e}")))?;
    Ok((manifest, vec![shader.bytes]))
}

impl Gpu {
    /// How long the calibration kernel takes on this machine's GPU, in nanoseconds: the
    /// fastest of 8 runs after `CALIBRATION_WARM_MS` of them to warm it (the GPU's clock at its
    /// height, where the guard's unit was measured), each a dispatch submitted and waited for
    /// alone (a few ms each), on a device of its own.
    pub(crate) fn calibration_ns() -> Result<f64> {
        CALIBRATING.store(true, Ordering::SeqCst);
        let timed = (|| {
            let (manifest, shaders) = calibration_kernel(CALIBRATION_CHAIN)?;
            let mut gpu = Gpu::new(&manifest, &shaders, Timing::Off, false)?;
            let n = CALIBRATION_GROUPS * 64;
            gpu.create_buffer(1, 4 * n);
            let mut times = Vec::new();
            let warming = std::time::Instant::now();
            let mut run = 0;
            // Warmed for at least `CALIBRATION_WARM_MS` (a GPU that sat idle lowers its clock,
            // and raises it under load), then timed 8 times.
            while times.len() < 8 {
                let began = std::time::Instant::now();
                gpu.dispatch(
                    0,
                    Ok([CALIBRATION_GROUPS, 1, 1]),
                    &[Binding::range(1, 0, 4 * n)],
                    &[],
                )?;
                gpu.flush()?;
                wait(&gpu.device)?;
                run += 1;
                if warming.elapsed().as_secs_f64() * 1000.0 >= CALIBRATION_WARM_MS || run > 400 {
                    times.push(began.elapsed().as_secs_f64() * 1e9);
                }
            }
            gpu.check_errors()?;
            Ok(times.into_iter().fold(f64::INFINITY, f64::min))
        })();
        CALIBRATING.store(false, Ordering::SeqCst);
        timed
    }
}

/// The guard's unit measured on this machine's GPU: the calibration kernel's fastest of up to
/// three timings (a GPU whose clock is still low reads slow), over the reference machine's,
/// snapped (`crate::step::snapped`); with the ratio and the time (ns).
pub(crate) fn measured_unit() -> Result<(u32, f64, f64)> {
    let limits = wrela_abi::Target::HERE.limits();
    let reference = f64::from(limits.calibration_ns.max(1));
    let mut ns = Gpu::calibration_ns()?;
    for _ in 0..2 {
        if ns / reference <= crate::step::UNIT_SLACK {
            break;
        }
        ns = ns.min(Gpu::calibration_ns()?);
    }
    let ratio = ns / reference;
    Ok((crate::step::snapped(limits.step_femtoseconds, ratio), ratio, ns))
}

/// Sets the guard's unit for the device `info` names, once a process (`crate::step`, item 39):
/// `WRELA_GPU_UNIT_FS` where it's set, else the one kept for this device and driver, else
/// measured (`Gpu::calibration_ns`), snapped and kept. Says which on stderr.
pub(super) fn ensure_unit(info: &wrela_gpu::AdapterInfo) -> Result<()> {
    if crate::step::unit_set() || CALIBRATING.load(Ordering::SeqCst) {
        return Ok(());
    }
    let limits = wrela_abi::Target::HERE.limits();
    let target = limits.step_femtoseconds;
    if let Some(fs) = std::env::var(crate::step::UNIT_VAR).ok().and_then(|v| v.trim().parse().ok())
    {
        crate::step::set_unit_fs(fs);
        eprintln!(
            "wrela-host: the guard's unit is {fs} fs a unit of work (forced: {})",
            crate::step::UNIT_VAR
        );
        return Ok(());
    }
    if limits.calibration_ns == 0 {
        crate::step::set_unit_fs(target);
        return Ok(());
    }
    let path = crate::step::unit_path(info);
    let kept = path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| t.split_whitespace().next().and_then(|w| w.parse::<u32>().ok()))
        .filter(|&fs| crate::step::is_snapped(target, fs));
    if let Some(fs) = kept {
        crate::step::set_unit_fs(fs);
        eprintln!(
            "wrela-host: the guard's unit is {fs} fs a unit of work (read from {})",
            path.as_ref().map_or(String::new(), |p| p.display().to_string())
        );
        return Ok(());
    }
    let (fs, ratio, ns) = measured_unit()?;
    crate::step::set_unit_fs(fs);
    if let Some(p) = &path {
        let _ = std::fs::create_dir_all(p.parent().unwrap_or(p));
        let _ = std::fs::write(p, format!("{fs} {ratio:.3} {ns:.0}\n"));
    }
    eprintln!(
        "wrela-host: the guard's unit is {fs} fs a unit of work (measured: the calibration kernel {:.3} ms, {ratio:.2} times the reference machine's; kept in {})",
        ns / 1e6,
        path.as_ref().map_or(String::new(), |p| p.display().to_string())
    );
    Ok(())
}
