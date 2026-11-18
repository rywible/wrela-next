//! The program's voice played on the machine's audio device (cpal: Core Audio on macOS, ALSA
//! on Linux), as the browser host plays it in an AudioWorklet: the device's callback renders
//! the voice's quanta as it needs them, on its own thread, on the voice's own instance
//! ([`crate::program::Voice`]), and writes them out. The offline render
//! ([`crate::CpuHost::render_audio`]) stays the test path; a voice played live gives the same
//! samples (toolchain/tests's `the_live_voice_is_the_offline_voice`).
//!
//! The voice is 48 kHz, two channels. The device plays at its own rate, which the host doesn't
//! change (on macOS that would change it for every program) and doesn't yet resample from: a
//! device that isn't at 48 kHz plays nothing, and says why. A device of more channels gets the
//! voice in its first two.
//!
//! The callback never waits: it takes the voice's lock only if it's free (the host takes it only
//! to start and stop the device, and reads how it goes from counters), and writes silence else,
//! counted. Each callback's time is kept,
//! against the time its buffer plays: a callback that takes longer is a glitch a person hears.

use crate::Error;
use crate::program::Voice;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The voice's rate (Hz).
const RATE: u32 = wrela_abi::AUDIO_SAMPLE_RATE;

/// What the host reads of the callback while it plays, without its lock.
#[derive(Default)]
struct Counters {
    /// Callbacks that found the voice's lock taken.
    busy: AtomicU64,
    tapped: AtomicUsize,
    trapped: AtomicBool,
}

/// What the device's callback keeps.
struct Shared {
    voice: Option<Voice>,
    /// The quantum being played, and the next sample of it to play (`SAMPLES`: none left).
    carry: [f32; Voice::SAMPLES],
    at: usize,
    /// The samples written, two channels interleaved, up to `tap_cap` (room reserved for them).
    tap: Vec<f32>,
    tap_cap: usize,
    trap: Option<wasmtime::Error>,
    callbacks: u64,
    quanta: u64,
    frames: (usize, usize),
    worst: f64,
    over: u64,
}

impl Shared {
    /// The next quantum into `carry`: silence once the voice has trapped.
    fn render(&mut self) {
        let Shared { voice, carry, trap, quanta, .. } = self;
        match voice.as_mut().filter(|_| trap.is_none()) {
            Some(v) => match v.quantum(carry) {
                Ok(()) => *quanta += 1,
                Err(e) => {
                    *trap = Some(e);
                    carry.fill(0.0);
                }
            },
            None => carry.fill(0.0),
        }
        self.at = 0;
    }

    /// Fills `out`, frames of `channels` samples, from the voice.
    fn fill(&mut self, out: &mut [f32], channels: usize) {
        for frame in out.chunks_exact_mut(channels) {
            if self.at == Voice::SAMPLES {
                self.render();
            }
            let (l, r) = (self.carry[self.at], self.carry[self.at + 1]);
            self.at += 2;
            frame[0] = l;
            frame[1] = r;
            frame[2..].fill(0.0);
            // Within the room reserved: the callback never allocates.
            if self.tap.len() + 2 <= self.tap_cap {
                self.tap.push(l);
                self.tap.push(r);
            }
        }
    }
}

/// The voice playing on a device.
pub struct Playing {
    stream: cpal::Stream,
    shared: Arc<Mutex<Shared>>,
    counters: Arc<Counters>,
    errors: Arc<Mutex<Vec<String>>>,
    /// The program whose voice it is.
    pub(crate) program: u64,
    /// The device, and its channels.
    pub device: String,
    pub channels: u16,
}

/// What a voice played: the samples written (as many as were asked for), and how the device's
/// callbacks went.
#[derive(Clone, Debug, Default)]
pub struct Played {
    pub device: String,
    pub channels: u16,
    /// The samples the callbacks wrote, the two channels interleaved, from the first.
    pub tap: Vec<f32>,
    pub callbacks: u64,
    pub quanta: u64,
    /// The fewest and most frames a callback asked for.
    pub frames: (usize, usize),
    /// The slowest callback's time as a share of the time its buffer plays, and the callbacks
    /// that took longer than it (each a glitch).
    pub worst: f64,
    pub over: u64,
    /// Callbacks that found the voice's lock taken, and wrote silence.
    pub busy: u64,
    /// What the device said went wrong.
    pub errors: Vec<String>,
}

impl Playing {
    /// Samples written so far, up to what was asked for.
    pub fn tapped(&self) -> usize {
        self.counters.tapped.load(Ordering::Relaxed)
    }

    /// Whether the voice has trapped (it plays silence from then on).
    pub fn trapped(&self) -> bool {
        self.counters.trapped.load(Ordering::Relaxed)
    }

    /// Stops the device: the voice back (if it didn't trap), its trap, and what it played.
    pub(crate) fn stop(self) -> (Option<Voice>, Option<wasmtime::Error>, Played) {
        drop(self.stream);
        let mut s = self.shared.lock().unwrap_or_else(|p| p.into_inner());
        let played = Played {
            device: self.device,
            channels: self.channels,
            tap: std::mem::take(&mut s.tap),
            callbacks: s.callbacks,
            quanta: s.quanta,
            frames: if s.callbacks == 0 { (0, 0) } else { s.frames },
            worst: s.worst,
            over: s.over,
            busy: self.counters.busy.load(Ordering::Relaxed),
            errors: std::mem::take(&mut *self.errors.lock().unwrap_or_else(|p| p.into_inner())),
        };
        (s.voice.take(), s.trap.take(), played)
    }
}

/// Plays `voice` (program `program`'s) on the machine's default output device, keeping the
/// first `tap` samples it writes. On failure, the voice back with why.
pub(crate) fn play(
    voice: Voice,
    program: u64,
    tap: usize,
) -> std::result::Result<Playing, Box<(Voice, Error)>> {
    let no = |why: String| Error::Program(format!("no sound: {why}"));
    let host = cpal::default_host();
    let Some(device) = host.default_output_device() else {
        return Err(Box::new((voice, no("the machine has no audio output device".into()))));
    };
    let name = device.to_string();
    let config = match output_config(&device) {
        Ok(c) => c,
        Err(why) => return Err(Box::new((voice, no(format!("{name}: {why}"))))),
    };
    let channels = config.channels;
    let shared = Arc::new(Mutex::new(Shared {
        voice: Some(voice),
        carry: [0.0; Voice::SAMPLES],
        at: Voice::SAMPLES,
        tap: Vec::with_capacity(tap),
        tap_cap: tap,
        trap: None,
        callbacks: 0,
        quanta: 0,
        frames: (usize::MAX, 0),
        worst: 0.0,
        over: 0,
    }));
    let counters = Arc::new(Counters::default());
    let errors = Arc::new(Mutex::new(Vec::new()));
    let (s, c, e) = (shared.clone(), counters.clone(), errors.clone());
    let n = usize::from(channels);
    let data = move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
        let began = Instant::now();
        let Ok(mut s) = s.try_lock() else {
            out.fill(0.0);
            c.busy.fetch_add(1, Ordering::Relaxed);
            return;
        };
        s.fill(out, n);
        c.tapped.store(s.tap.len(), Ordering::Relaxed);
        c.trapped.store(s.trap.is_some(), Ordering::Relaxed);
        let frames = out.len() / n;
        let share = began.elapsed().as_secs_f64() * f64::from(RATE) / frames.max(1) as f64;
        s.callbacks += 1;
        s.frames = (s.frames.0.min(frames), s.frames.1.max(frames));
        s.worst = s.worst.max(share);
        s.over += u64::from(share > 1.0);
    };
    let error = move |err: cpal::Error| {
        if let Ok(mut e) = e.lock() {
            e.push(err.to_string());
        }
    };
    let stream = match device.build_output_stream::<f32, _, _>(config, data, error, None) {
        Ok(s) => s,
        Err(why) => return Err(Box::new((take_back(&shared), no(format!("{name}: {why}"))))),
    };
    if let Err(why) = stream.play() {
        drop(stream);
        return Err(Box::new((take_back(&shared), no(format!("{name}: {why}")))));
    }
    Ok(Playing { stream, shared, counters, errors, program, device: name, channels })
}

fn take_back(shared: &Mutex<Shared>) -> Voice {
    let mut s = shared.lock().unwrap_or_else(|p| p.into_inner());
    s.voice.take().expect("the voice was put there")
}

/// The device's stream at 48 kHz, `f32`, at least two channels, without changing the device's
/// own rate: its default if that's it, else one of its others at the same rate and channels.
fn output_config(device: &cpal::Device) -> std::result::Result<cpal::StreamConfig, String> {
    let default = device.default_output_config().map_err(|e| e.to_string())?;
    if default.sample_rate() != RATE {
        return Err(format!(
            "it plays at {} Hz, and the voice is {RATE} Hz: the host doesn't resample yet, nor \
             change a device's rate",
            default.sample_rate()
        ));
    }
    if default.channels() < 2 {
        return Err(format!("it has {} channel(s); the voice has two", default.channels()));
    }
    let f32_ok = default.sample_format() == cpal::SampleFormat::F32
        || device.supported_output_configs().map_err(|e| e.to_string())?.any(|c| {
            c.sample_format() == cpal::SampleFormat::F32
                && c.channels() == default.channels()
                && c.try_with_sample_rate(RATE).is_some()
        });
    if !f32_ok {
        return Err(format!("it takes no f32 samples at {RATE} Hz (its own: {default:?})"));
    }
    Ok(cpal::StreamConfig {
        channels: default.channels(),
        sample_rate: RATE,
        buffer_size: cpal::BufferSize::Default,
    })
}
