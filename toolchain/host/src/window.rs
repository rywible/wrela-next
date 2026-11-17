//! `wrela-host <build> --window`: the program's frames shown in a window as they're drawn
//! (winit), until it's closed or `--frames` have run. Each frame is a lockstep frame, as a
//! headless run's (its ticks, then the frame, at `--fps`), drawn into the screen and then into
//! the window's next frame ([`crate::Host::present`]), which shows it as `--present` says: `fifo`
//! (each refresh, no tearing), `fifo-relaxed`, `mailbox` (the newest at each refresh) or
//! `immediate` (at once, may tear). The window's size is the screen's: resizing it resizes the
//! frames the program draws.
//!
//! Frames are paced ([`crate::pace`]): each takes its input as late as it can and still make the
//! refresh it aims at, one frame a refresh. The loop draws between events, never inside one, so
//! the events that come while a frame waits for its time reach that frame.

use crate::pace::{Pacer, Plan};
use crate::{Error, Host, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// How a window run goes.
pub struct WindowRun {
    /// Frames to run before closing (`None`: until the window is closed).
    pub frames: Option<u32>,
    pub fps: f64,
    pub size: (u32, u32),
    pub present: wrela_gpu::PresentMode,
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
    pacer: Pacer,
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

    /// Runs the next frame and shows it, aiming at `plan`'s refresh.
    fn draw(&mut self, plan: Plan, el: &ActiveEventLoop) {
        let (w, h) = self.size;
        // The frame's input is what's come by now: its latency runs from here.
        let sampled = self.now();
        let drawn = self
            .host
            .lockstep_frame(self.frame, self.run.fps, w, h, &[])
            .and_then(|()| self.host.present());
        match drawn {
            Ok(Some(id)) => {
                let presented = self.now();
                self.pacer.presented(id, sampled, presented, plan.target);
                self.waiting.insert(id, (sampled, presented));
            }
            Ok(None) => {}
            Err(e) => return self.fail(e, el),
        }
        self.frame += 1;
        self.drawn_at = self.now();
        if self.run.frames.is_some_and(|n| self.frame >= n) {
            el.exit();
        }
    }

    fn fail(&mut self, e: Error, el: &ActiveEventLoop) {
        self.failed = Some(e);
        el.exit();
    }
}

impl ApplicationHandler for App<'_> {
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
            std::thread::sleep(Duration::from_secs_f64(early));
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

    /// Frames shown a second after the first.
    pub fn rate(&self) -> Option<f64> {
        let (first, last) = (self.shown.get(1)?, self.shown.last()?);
        let span = last - first;
        (span > 0.0).then(|| (self.shown.len() - 2) as f64 / span)
    }
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
    let event_loop = EventLoop::new().map_err(|e| Error::Gpu(format!("no event loop: {e}")))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        host,
        run,
        window: None,
        size: run.size,
        frame: 0,
        failed: None,
        start: Instant::now(),
        pacer: Pacer::new(1.0 / 60.0),
        trace: std::env::var_os("WRELA_PACE_TRACE").is_some(),
        drawn_at: 0.0,
        waiting: HashMap::new(),
        latencies: Vec::new(),
        before_present: Vec::new(),
        shown: Vec::new(),
    };
    event_loop.run_app(&mut app).map_err(|e| Error::Gpu(format!("the window: {e}")))?;
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
    })
}
