//! `wrela-host <build> --window`: the program's frames shown in a window as they're drawn
//! (winit), until it's closed or `--frames` have run, on play's schedule ([`crate::paced`]): the
//! program's ticker on its own thread and clock, and each frame at the time it takes its input
//! (seconds since the first frame was drawn), drawn into the screen and then into the window's
//! next frame ([`crate::Host::present`]), which shows it as `--present` says: `fifo` (each
//! refresh, no tearing), `fifo-relaxed`, `mailbox` (the newest at each refresh) or `immediate`
//! (at once, may tear). The window's size is the screen's: resizing it resizes the frames the
//! program draws. (A headless run is test mode's lockstep instead: frame `i` at `i / fps`.)
//!
//! Frames are paced ([`crate::pace`]): each takes its input as late as it can and still make the
//! refresh it aims at, one frame a refresh. The loop draws between events, never inside one, and
//! waits out the last moment before a frame awake, so the events that come while a frame waits
//! for its time reach that frame: a person's keys, pointer, wheel and gamepads ([`crate::live`]),
//! and a script's events at that frame, if the run has one. Each frame hands what it took to the
//! ticker, whose next tick takes it as records.
//!
//! The program's voice, once it starts one, plays on the machine's audio device
//! ([`crate::voice`]); a machine with none it can play on runs silent, and says why.
//!
//! What it checks (the parry-tick test): with a tick log ([`WindowRun::log`]), that each event the
//! frames took reached the sim in order, on the first tick that started after its frame handed
//! it over ([`check_ticks`]), and how long that took; and a script's events can arrive as a
//! person's do ([`WindowRun::arrive`]), each at a time on the clock, through the event loop, so
//! that an event taken a frame later than the first that sampled after it came, a frame of
//! latency lost, is counted ([`WindowRan::arrivals`]). The log replays the sim
//! (`wrela-host --replay`).

use crate::live::Live;
use crate::pace::{Pacer, Plan};
use crate::paced::{Ran, Ticker};
use crate::voice::{Played, Playing};
use crate::{Error, Host, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};
use wrela_abi::input::{At, Event, EventKind, Scripted};
use wrela_abi::memory::MAX_TICK_RECORDS;
use wrela_abi::ticks::{Tick, TickLog};

/// How a window run goes.
pub struct WindowRun {
    /// Frames to run before closing (`None`: until the window is closed).
    pub frames: Option<u32>,
    /// The rate `arrive`'s frames are counted at (frames a second).
    pub fps: f64,
    pub size: (u32, u32),
    pub present: wrela_gpu::PresentMode,
    /// Events that arrive before given frames, besides a person's, and records of given ticks
    /// (`--input`).
    pub script: Vec<Scripted>,
    /// Events that arrive as a person's do (`--arrive`): the one at frame `n` (the `k`th) at
    /// `(n + frac(k·φ)) / fps` seconds after the first frame is drawn, the fractions spreading
    /// them over every moment of a frame.
    pub arrive: Vec<Scripted>,
    /// Keep what each frame took ([`WindowRan::taken`]).
    pub keep: bool,
    /// Keep a tick log ([`WindowRan::log`]): the program's ticks, their records and hashes.
    pub log: bool,
}

/// An event that arrived as a person's does, and when it was sent.
struct Arrival {
    event: Event,
    sent: Instant,
}

struct App<'a> {
    host: &'a mut Host,
    run: &'a WindowRun,
    window: Option<Arc<Window>>,
    size: (u32, u32),
    frame: u32,
    failed: Option<Error>,
    /// The run's clock: seconds since this.
    start: Instant,
    /// Play's clock: frame time 0, when the first frame was drawn.
    play: Option<Instant>,
    /// The program that runs: another build taking its place ends the run.
    program: u64,
    pacer: Pacer,
    /// The device's epoch at the last frame: a frame in which it changed was a device made
    /// again (a hitch).
    epoch: u32,
    live: Live,
    ticker: Option<Ticker>,
    /// The voice playing, once the program starts it; why it can't, if it can't.
    voice: Option<Playing>,
    silent: Option<String>,
    /// The events each frame took, in order, and when the ticker was handed them (seconds;
    /// before the ticker started, its start), if kept.
    taken: Vec<(Scripted, f64)>,
    log: Option<TickLog>,
    /// When each tick took its records (seconds), and the ticks dropped.
    took: Vec<f64>,
    dropped: u32,
    /// Sends `--arrive`'s events, once the first frame is drawn.
    proxy: Option<EventLoopProxy<Arrival>>,
    /// When each event that arrived and isn't taken yet was sent (seconds).
    arriving: Vec<f64>,
    /// When each one that was taken was sent, and the frame that took it.
    arrived: Vec<(f64, u32)>,
    /// When each frame sampled its input (seconds).
    sampled: Vec<f64>,
    /// `WRELA_PACE_TRACE`: each plan and each frame shown, to stderr.
    trace: bool,
    /// When the last frame was drawn: a window that draws none for a second has stalled.
    drawn_at: f64,
    /// When each frame not yet shown took its input and was presented, by its present's id.
    waiting: HashMap<u64, (f64, f64)>,
    /// Each shown frame's latency, input sampled to photons, and of it input to presented (the
    /// frame run, recorded and submitted): milliseconds; and when it was shown (seconds).
    latencies: Vec<f64>,
    before_present: Vec<f64>,
    shown: Vec<f64>,
}

impl App<'_> {
    fn now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn secs(&self, at: Instant) -> f64 {
        at.saturating_duration_since(self.start).as_secs_f64()
    }

    /// The frames shown since the last look: their latencies, and the pacer told.
    fn note_shown(&mut self) {
        for (id, at) in self.host.take_presented() {
            let at = self.secs(at);
            if self.trace {
                eprintln!(
                    "pace: frame {id} shown at {:.1} ms (told at {:.1})",
                    at * 1e3,
                    self.now() * 1e3
                );
            }
            self.pacer.shown(id, at);
            // Frames are shown in order: one before this not shown yet was overtaken.
            self.waiting.retain(|&k, _| k >= id);
            if let Some((sampled, presented)) = self.waiting.remove(&id) {
                self.latencies.push((at - sampled) * 1e3);
                self.before_present.push((presented - sampled) * 1e3);
                self.shown.push(at);
            }
        }
    }

    /// What the ticker did since the last look: the log kept, a trap the run's failure.
    fn note_ticks(&mut self, ran: Vec<Ran>) -> Result<()> {
        for r in ran {
            match r {
                Ran::Tick { ticked, took } => {
                    let took = self.secs(took);
                    self.took.push(took);
                    if let (Some(log), Some(hash)) = (&mut self.log, ticked.hash) {
                        log.ticks.push(Tick { records: ticked.records, hash });
                    }
                }
                Ran::Dropped { ticks, .. } => self.dropped += ticks,
                Ran::Trapped { tick, trap } => return Err(self.host.tick_failed(tick, &trap)),
            }
        }
        Ok(())
    }

    /// Runs the next frame and shows it, aiming at `plan`'s refresh.
    fn draw(&mut self, plan: Plan, el: &ActiveEventLoop) {
        if let Err(e) = self.frame_once(plan) {
            return self.fail(e, el);
        }
        self.frame += 1;
        self.drawn_at = self.now();
        if self.run.frames.is_some_and(|n| self.frame >= n) {
            el.exit();
        }
    }

    fn frame_once(&mut self, plan: Plan) -> Result<()> {
        let (w, h) = self.size;
        // The frame's input is what's come by now: its latency runs from here.
        let sampled_at = Instant::now();
        let sampled = self.secs(sampled_at);
        let at = At::Frame(self.frame);
        let mut events: Vec<Scripted> =
            self.run.script.iter().filter(|s| s.at == at).cloned().collect();
        events.extend(self.live.take().into_iter().map(|event| Scripted {
            at,
            event,
            payload: Vec::new(),
        }));
        self.sampled.push(sampled);
        let frame = self.frame;
        self.arrived.extend(self.arriving.drain(..).map(|sent| (sent, frame)));
        // The frame's input, and the ticker's records: after what the program posted, the
        // same events (a tick takes no network events). Before the ticker starts, they wait
        // in the program's queue for its first tick.
        for s in &events {
            self.host.push_scripted(s);
        }
        let ticking = self.host.ticker_hz().is_some();
        if ticking {
            self.host.push_records(
                events.iter().filter(|s| !s.event.kind.is_network()).map(|s| s.event),
            );
        }
        let handed = match &self.ticker {
            Some(t) => {
                let records = self.host.take_records();
                Some(self.secs(t.queue(records)))
            }
            None => None,
        };
        let time = self.play.map_or(0.0, |p| sampled_at.saturating_duration_since(p).as_secs_f64());
        self.host.frame(time as f32, w, h)?;
        if self.host.program_serial() != self.program {
            return Err(Error::Program(
                "another build took the program's place: a window doesn't hand over yet".into(),
            ));
        }
        // A device made again this frame (`Gpu::recover`): a hitch the pacer mustn't learn as
        // work (learned, every third frame missed its refresh for the 120 frames after a loss).
        let epoch = self.host.device_epoch();
        let made_again = epoch != self.epoch;
        self.epoch = epoch;
        if let Some(id) = self.host.present()? {
            let presented = self.now();
            if made_again {
                self.pacer.presented_after_hitch(id, presented, plan.target);
            } else {
                self.pacer.presented(id, sampled, presented, plan.target);
            }
            self.waiting.insert(id, (sampled, presented));
        }
        if self.run.keep {
            self.taken.extend(events.into_iter().map(|s| (s, handed.unwrap_or(f64::NAN))));
        }
        if self.frame == 0 {
            // Play's clock starts once the first frame (which builds the pipelines) is drawn.
            self.play = Some(Instant::now());
            self.start_arriving();
        }
        if self.ticker.is_none() && ticking {
            let play = self.play.expect("set at the first frame");
            let ticks = self.run.script.iter().filter(|s| matches!(s.at, At::Tick(_)));
            // What was waiting is handed to the ticker as it starts, for its first tick.
            let handed = self.now();
            self.ticker = self.host.start_ticker(play, ticks.cloned().collect())?;
            for (_, h) in self.taken.iter_mut().filter(|(_, h)| h.is_nan()) {
                *h = handed;
            }
        } else if !ticking {
            // No ticker: nothing takes records.
            self.host.take_records();
        }
        if let Some(t) = &self.ticker {
            let ran = t.take();
            self.note_ticks(ran)?;
        }
        if self.voice.is_none() && self.silent.is_none() && self.host.has_voice() {
            match self.host.play_voice(0) {
                Ok(p) => self.voice = Some(p),
                Err(e) => {
                    eprintln!("{e}");
                    self.silent = Some(e.to_string());
                }
            }
        }
        if self.voice.as_ref().is_some_and(Playing::trapped) {
            let playing = self.voice.take().expect("checked");
            self.host.stop_voice(playing)?;
        }
        Ok(())
    }

    fn fail(&mut self, e: Error, el: &ActiveEventLoop) {
        self.failed = Some(e);
        el.exit();
    }

    /// `--arrive`'s events, sent from a thread of their own at their times from now on.
    fn start_arriving(&mut self) {
        let Some(proxy) = self.proxy.take() else { return };
        let base = Instant::now();
        let fps = self.run.fps;
        // φ's fraction: each event lands at another moment of its frame, evenly over many.
        let phi = (5f64.sqrt() - 1.0) / 2.0;
        let times: Vec<(Duration, Event)> = self
            .run
            .arrive
            .iter()
            .enumerate()
            .filter_map(|(k, s)| match s.at {
                At::Frame(n) => {
                    let at = (f64::from(n) + (k as f64 * phi).fract()) / fps;
                    Some((Duration::from_secs_f64(at), s.event))
                }
                At::Tick(_) => None,
            })
            .collect();
        std::thread::spawn(move || {
            for (at, event) in times {
                let due = base + at;
                // Slept to within a millisecond, then waited out exactly.
                if let Some(rest) = due.checked_duration_since(Instant::now()) {
                    std::thread::sleep(rest.saturating_sub(Duration::from_millis(1)));
                }
                while Instant::now() < due {
                    std::hint::spin_loop();
                }
                if proxy.send_event(Arrival { event, sent: Instant::now() }).is_err() {
                    return;
                }
            }
        });
    }
}

impl ApplicationHandler<Arrival> for App<'_> {
    fn user_event(&mut self, _: &ActiveEventLoop, a: Arrival) {
        self.live.push(a.event);
        let sent = self.secs(a.sent);
        self.arriving.push(sent);
    }

    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let (w, h) = self.run.size;
        let attributes = Window::default_attributes()
            .with_title("wrela")
            .with_inner_size(winit::dpi::PhysicalSize::new(w, h));
        let window = match el.create_window(attributes) {
            Ok(w) => Arc::new(w),
            Err(e) => return self.fail(Error::Gpu(format!("no window: {e}")), el),
        };
        let size = window.inner_size();
        self.size = (size.width.max(1), size.height.max(1));
        let handle: Arc<dyn wrela_gpu::Window> = window.clone();
        if let Err(e) = self.host.attach_window(handle, self.size.0, self.size.1, self.run.present)
        {
            return self.fail(e, el);
        }
        self.window = Some(window);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        self.live.window_event(&event);
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(s) => {
                if s.width > 0 && s.height > 0 {
                    self.size = (s.width, s.height);
                    if let Err(e) = self.host.resize_window(s.width, s.height) {
                        self.fail(e, el);
                    }
                }
            }
            _ => {}
        }
    }

    /// The events that have come are handled: the next frame goes now if it's time, or the loop
    /// waits for its time (and for events meanwhile).
    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        if self.window.is_none() || self.failed.is_some() || el.exiting() {
            return;
        }
        self.note_shown();
        let now = self.now();
        // The first frame builds the program's pipelines: it may take seconds.
        if self.frame > 1 && now - self.drawn_at > 1.0 {
            let e = format!(
                "the window stalled: no frame drawn for {:.0} ms after frame {}; {}",
                (now - self.drawn_at) * 1e3,
                self.frame,
                self.pacer.describe()
            );
            return self.fail(Error::Gpu(e), el);
        }
        let plan = self.pacer.plan(now);
        if self.trace {
            eprintln!(
                "pace: at {:.1} ms, frame {} plans input at {:.1} aiming at {:?}; {}",
                now * 1e3,
                self.frame,
                plan.at * 1e3,
                plan.target.map(|t| (t * 1e4).round() / 10.0),
                self.pacer.describe()
            );
        }
        let early = plan.at - now;
        if early > 0.002 {
            // Woken a millisecond early (a wake can be late), to wait out the rest exactly.
            let wake = self.start + Duration::from_secs_f64(plan.at - 0.001);
            el.set_control_flow(ControlFlow::WaitUntil(wake));
            return;
        }
        if early > 0.0 {
            // The rest waited out by the loop itself, not asleep: an event that comes in the
            // last moment before the frame samples is still this frame's.
            el.set_control_flow(ControlFlow::Poll);
            return;
        }
        self.draw(plan, el);
        el.set_control_flow(ControlFlow::Poll);
    }
}

/// What a window run did: the frames it ran, and each shown frame's latency (ms, input
/// sampled to photons: when the frame took its input, just before it ran, to when the display
/// showed it, as Metal's `presentedTime` or Vulkan's `present_wait` says), in the order shown.
pub struct WindowRan {
    pub frames: u32,
    pub latencies: Vec<f64>,
    /// Of each, input sampled to presented: the frame run, recorded and submitted (ms).
    pub before_present: Vec<f64>,
    /// When each was shown (seconds).
    pub shown: Vec<f64>,
    /// Frames shown after the refresh they aimed at, or never.
    pub misses: u32,
    /// The display's refresh period as the pacer learned it (ms).
    pub period: f64,
    /// The pacer's state at the end ([`Pacer::describe`]): why the latency is what it is.
    pub pacer: String,
    /// The events each frame took (a person's and the script's), at their frames, and when the
    /// ticker was handed them (seconds), if kept ([`WindowRun::keep`]).
    pub taken: Vec<(Scripted, f64)>,
    /// The tick log, if kept ([`WindowRun::log`]), and when each tick in it took its records
    /// (seconds).
    pub log: Option<TickLog>,
    pub took: Vec<f64>,
    /// Ticks the ticker dropped, behind its clock ([`crate::paced::BEHIND`]).
    pub dropped: u32,
    /// How the voice played, if the program started one and the machine could play it; or why
    /// it couldn't.
    pub voice: Option<Played>,
    pub silent: Option<String>,
    /// When each frame sampled its input (seconds), and of the events that arrived as a
    /// person's ([`WindowRun::arrive`]), when each was sent and which frame took it.
    pub sampled: Vec<f64>,
    pub arrived: Vec<(f64, u32)>,
}

/// How the events that arrived as a person's were taken ([`WindowRan::arrivals`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arrivals {
    pub count: usize,
    /// From arriving to being taken (ms): the median, and the worst.
    pub median: f64,
    pub worst: f64,
    /// Taken a frame (or more) later than the first frame that sampled more than [`LATE`]
    /// after they arrived: their count, and the most a frame sampled after one of them and
    /// didn't take it (ms).
    pub late: usize,
    pub late_by: f64,
}

/// How long after an event arrives a frame may sample without taking it (s): the time the
/// event loop takes to hand it over.
pub const LATE: f64 = 0.0005;

/// How the events the frames took reached the sim ([`check_ticks`]): how many, and from their
/// frame handing them to the ticker to the tick that took them (ms), the median and the worst.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reached {
    pub count: usize,
    pub median: f64,
    pub worst: f64,
}

impl WindowRan {
    /// The first frame shown's latency (ms): it waits for the program's pipelines to be built.
    pub fn first(&self) -> Option<f64> {
        self.latencies.first().copied()
    }

    /// The latencies' median, 90th percentile and worst (ms) after the first, or `None` with
    /// none.
    pub fn latency(&self) -> Option<(f64, f64, f64)> {
        stats(self.latencies.get(1..)?)
    }

    /// The same, of the second half of the run: once the pacer has settled.
    pub fn settled(&self) -> Option<(f64, f64, f64)> {
        stats(self.latencies.get(self.latencies.len() / 2..)?)
    }

    /// The same of input sampled to presented.
    pub fn before_present(&self) -> Option<(f64, f64, f64)> {
        stats(self.before_present.get(1..)?)
    }

    /// How the events that arrived as a person's were taken, or `None` with none.
    pub fn arrivals(&self) -> Option<Arrivals> {
        arrivals(&self.arrived, &self.sampled)
    }

    /// Frames shown a second, over the second half of those shown (as `settled`): a pause
    /// before it (the first frames, a device made again) isn't the settled rate.
    pub fn rate(&self) -> Option<f64> {
        let half = self.shown.get(self.shown.len() / 2..)?;
        let (first, last) = (half.first()?, half.last()?);
        let span = last - first;
        (span > 0.0 && half.len() > 1).then(|| (half.len() - 1) as f64 / span)
    }
}

/// How events sent at given times (seconds) and taken by given frames were taken, given when
/// each frame sampled its input.
fn arrivals(arrived: &[(f64, u32)], sampled: &[f64]) -> Option<Arrivals> {
    let delays: Vec<f64> =
        arrived.iter().map(|&(sent, f)| (sampled[f as usize] - sent) * 1e3).collect();
    let (median, _, worst) = stats(&delays)?;
    // A frame before the one that took it sampled after it arrived.
    let missed: Vec<f64> = arrived
        .iter()
        .filter_map(|&(sent, f)| {
            let before = sampled[..f as usize].last()?;
            (*before > sent + LATE).then(|| (before - sent) * 1e3)
        })
        .collect();
    Some(Arrivals {
        count: arrived.len(),
        median,
        worst,
        late: missed.len(),
        late_by: missed.iter().copied().fold(0.0, f64::max),
    })
}

/// Whether each event the frames took (`taken`, in order, each with when its frame handed it to
/// the ticker) reached the sim on the tick it should: in the order taken, a record of the first
/// tick that took its records after the event was handed over (`took`: when each tick in `log`
/// did), unless that tick was full (`MAX_TICK_RECORDS`). A program's own posts in the log are
/// passed over, and events handed over after the last tick took its records aren't records
/// yet. How they reached it, or the first event that went elsewhere.
pub fn check_ticks(
    log: &TickLog,
    took: &[f64],
    taken: &[(Scripted, f64)],
) -> std::result::Result<Reached, String> {
    let posted = (EventKind::Posted as u32).to_le_bytes();
    let mut records = log.ticks.iter().enumerate().flat_map(|(t, tick)| {
        tick.records.iter().filter(|r| r[..4] != posted).map(move |r| (t, r))
    });
    let mut delays = Vec::new();
    for (n, (s, handed)) in taken.iter().enumerate() {
        if s.event.kind.is_network() {
            continue;
        }
        let what = |s: &Scripted| match s.at {
            At::Frame(k) => format!("event {n} ({:?}, taken by frame {k})", s.event.kind),
            At::Tick(k) => format!("event {n} ({:?}, tick {k}'s)", s.event.kind),
        };
        let Some((t, record)) = records.next() else {
            if took.get(log.ticks.len().wrapping_sub(1)).is_none_or(|last| handed > last) {
                break; // handed over after the last tick took its records
            }
            return Err(format!("{} is in no tick's records", what(s)));
        };
        if *record != s.event.bytes() {
            return Err(format!(
                "{} isn't tick {t}'s next record: the ticks took another, or in another order",
                what(s)
            ));
        }
        let ms = |secs: f64| secs * 1e3;
        if took[t] < *handed {
            return Err(format!(
                "{} is tick {t}'s, which took its records at {:.3} ms, before it was handed over \
                 at {:.3} ms",
                what(s),
                ms(took[t]),
                ms(*handed)
            ));
        }
        if t > 0
            && took[t - 1] >= *handed
            && log.ticks[t - 1].records.len() < MAX_TICK_RECORDS as usize
        {
            return Err(format!(
                "{} reached tick {t}, but tick {} took its records after it was handed over \
                 ({:.3} ms, at {:.3} ms) and had room",
                what(s),
                t - 1,
                ms(took[t - 1]),
                ms(*handed)
            ));
        }
        delays.push(ms(took[t] - handed));
    }
    let (median, _, worst) = stats(&delays).unwrap_or_default();
    Ok(Reached { count: delays.len(), median, worst })
}

/// Median, 90th percentile and worst.
fn stats(l: &[f64]) -> Option<(f64, f64, f64)> {
    let mut l = l.to_vec();
    if l.is_empty() {
        return None;
    }
    l.sort_by(f64::total_cmp);
    let at = |q: f64| l[((l.len() - 1) as f64 * q).round() as usize];
    Some((at(0.5), at(0.9), l[l.len() - 1]))
}

/// Runs `host`'s frames in a window as `run` says.
pub fn run(host: &mut Host, run: &WindowRun) -> Result<WindowRan> {
    let event_loop = EventLoop::<Arrival>::with_user_event()
        .build()
        .map_err(|e| Error::Gpu(format!("no event loop: {e}")))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let log = if run.log { Some(host.tick_log()?) } else { None };
    let proxy = (!run.arrive.is_empty()).then(|| event_loop.create_proxy());
    let host_program = host.program_serial();
    let mut app = App {
        host,
        run,
        window: None,
        size: run.size,
        frame: 0,
        failed: None,
        start: Instant::now(),
        play: None,
        program: host_program,
        pacer: Pacer::new(1.0 / 60.0),
        epoch: 0,
        live: Live::new(),
        ticker: None,
        voice: None,
        silent: None,
        taken: Vec::new(),
        log,
        took: Vec::new(),
        dropped: 0,
        proxy,
        arriving: Vec::new(),
        arrived: Vec::new(),
        sampled: Vec::new(),
        trace: std::env::var_os("WRELA_PACE_TRACE").is_some(),
        drawn_at: 0.0,
        waiting: HashMap::new(),
        latencies: Vec::new(),
        before_present: Vec::new(),
        shown: Vec::new(),
    };
    let ran = event_loop.run_app(&mut app);
    // The ticker stopped (after the tick it's running), its instance back with the program.
    if let Some(t) = app.ticker.take() {
        let ticks = app.host.stop_ticker(t);
        if let Err(e) = app.note_ticks(ticks) {
            app.failed.get_or_insert(e);
        }
    }
    let mut voice = None;
    if let Some(p) = app.voice.take() {
        match app.host.stop_voice(p) {
            Ok(played) => voice = Some(played),
            Err(e) => {
                app.failed.get_or_insert(e);
            }
        }
    }
    ran.map_err(|e| Error::Gpu(format!("the window: {e}")))?;
    if let Some(e) = app.failed {
        return Err(e);
    }
    // The last frames' photons come after the loop: what's shown within a moment counts.
    let until = Instant::now() + Duration::from_millis(250);
    while !app.waiting.is_empty() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(5));
        app.note_shown();
    }
    Ok(WindowRan {
        frames: app.frame,
        latencies: app.latencies,
        before_present: app.before_present,
        shown: app.shown,
        misses: app.pacer.misses,
        period: app.pacer.period() * 1e3,
        pacer: app.pacer.describe(),
        taken: app.taken,
        log: app.log,
        took: app.took,
        dropped: app.dropped,
        voice,
        silent: app.silent,
        sampled: app.sampled,
        arrived: app.arrived,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use wrela_abi::input::key_code;

    fn at(frame: u32, event: Event) -> Scripted {
        Scripted { at: At::Frame(frame), event, payload: Vec::new() }
    }

    /// A ticker's log as play's schedule makes it: each tick takes, when it starts, every record
    /// handed over before (at most `cap`). Ticks start every 10 ms from 5 ms.
    fn ticked(taken: &[(Scripted, f64)], ticks: usize, cap: usize) -> (TickLog, Vec<f64>) {
        let mut log = TickLog::new(0, 60, 0);
        let took: Vec<f64> = (0..ticks).map(|k| 0.005 + 0.010 * k as f64).collect();
        let mut queue = taken.iter().peekable();
        let mut waiting = Vec::new();
        for &t in &took {
            while let Some((s, _)) = queue.next_if(|(_, h)| *h <= t) {
                waiting.push(s.event.bytes());
            }
            let n = waiting.len().min(cap);
            log.ticks.push(Tick { records: waiting.drain(..n).collect(), hash: 0 });
        }
        (log, took)
    }

    #[test]
    fn each_event_reaches_the_first_tick_after_it_was_handed_over() {
        let space = |down| Event::key(down, key_code("Space"), false, 0);
        // Frames hand events over at 1, 12, 12 and 33 ms: ticks 0 (5 ms), 1 (15), 1 and 3 (35).
        let taken = vec![
            (at(0, space(true)), 0.001),
            (at(1, space(false)), 0.012),
            (at(1, Event::text('x')), 0.012),
            (at(3, space(true)), 0.033),
        ];
        let (log, took) = ticked(&taken, 5, 256);
        let reached = check_ticks(&log, &took, &taken).unwrap();
        assert_eq!(reached.count, 4);
        assert!((reached.worst - 4.0).abs() < 1e-6, "{reached:?}");
        // Tick 1's records a tick late: tick 1 took its records after they were handed over.
        let mut late = log.clone();
        let moved = std::mem::take(&mut late.ticks[1].records);
        late.ticks[2].records.splice(0..0, moved);
        let e = check_ticks(&late, &took, &taken).unwrap_err();
        assert!(e.contains("event 1") && e.contains("tick 1 took its records after"), "{e}");
        // Unless the tick was full: 257 handed over at once, the last a tick later.
        let many: Vec<_> = (0..257).map(|_| (at(1, Event::text('y')), 0.012)).collect();
        let (full, took_full) = ticked(&many, 3, MAX_TICK_RECORDS as usize);
        assert_eq!(full.ticks[2].records.len(), 1);
        assert_eq!(check_ticks(&full, &took_full, &many).unwrap().count, 257);
        // Out of order, or missing.
        let mut swapped = log.clone();
        swapped.ticks[1].records.swap(0, 1);
        assert!(check_ticks(&swapped, &took, &taken).unwrap_err().contains("another order"));
        let mut lost = log.clone();
        lost.ticks[3].records.clear();
        assert!(check_ticks(&lost, &took, &taken).unwrap_err().contains("no tick's records"));
        // A record a tick took before it was handed over.
        let mut early = log.clone();
        let moved = std::mem::take(&mut early.ticks[3].records);
        early.ticks[2].records.extend(moved);
        assert!(check_ticks(&early, &took, &taken).unwrap_err().contains("before it was handed"));
        // Handed over after the last tick took its records: not yet a record, not an error.
        let (short, took_short) = ticked(&taken, 3, 256);
        assert_eq!(check_ticks(&short, &took_short, &taken).unwrap().count, 3);
        // The program's own posts are passed over.
        let mut posted = log.clone();
        posted.ticks[1].records.insert(0, Event::posted(1, 2, 3).bytes());
        assert_eq!(check_ticks(&posted, &took, &taken).unwrap().count, 4);
    }

    #[test]
    fn an_event_a_frame_sampled_after_and_didnt_take_is_late() {
        // Frames sampled at 0, 10 and 20 ms.
        let sampled = [0.000, 0.010, 0.020];
        // Sent at 1 ms, taken at 10: on time. At 9.8 ms, taken at 20: late by 0.2 ms, within
        // what the loop takes to hand it over. At 5 ms, taken at 20: late by 5 ms.
        let a = arrivals(&[(0.001, 1), (0.0098, 2), (0.005, 2)], &sampled).unwrap();
        assert_eq!(a.count, 3);
        assert_eq!(a.late, 1);
        assert!((a.late_by - 5.0).abs() < 1e-9, "{a:?}");
        assert!((a.worst - 15.0).abs() < 1e-9, "{a:?}");
        assert!((a.median - 10.2).abs() < 1e-9, "{a:?}");
        assert_eq!(arrivals(&[], &sampled), None);
    }
}
