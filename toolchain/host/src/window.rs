//! `wrela-host <build> --window`: the program's frames shown in a window as they're drawn
//! (winit), until it's closed or `--frames` have run. Each frame is a lockstep frame, as a
//! headless run's (its ticks, then the frame, at `--fps`), drawn into the screen and then into
//! the window's next frame ([`crate::Host::present`]), which shows it as `--present` says: `fifo`
//! (each refresh, no tearing), `fifo-relaxed`, `mailbox` (the newest at each refresh) or
//! `immediate` (at once, may tear). The window's size is the screen's: resizing it resizes the
//! frames the program draws.

use crate::{Error, Host, Result};
use std::sync::Arc;
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
    /// The most frames sampled and not yet shown: a frame waits for the one this many before it
    /// to reach the display before it takes its input (the latency it adds, a frame each).
    pub in_flight: u32,
}

struct App<'a> {
    host: &'a mut Host,
    run: &'a WindowRun,
    window: Option<Arc<Window>>,
    size: (u32, u32),
    frame: u32,
    failed: Option<Error>,
    /// When each frame shown sampled its input (just before it ran) and was presented, by its
    /// present's id.
    sampled: std::collections::HashMap<u64, (std::time::Instant, std::time::Instant)>,
    /// Each shown frame's latency, input sampled to photons, and of it input to presented (the
    /// frame run, recorded and submitted): milliseconds.
    latencies: Vec<f64>,
    before_present: Vec<f64>,
    /// When each was shown.
    shown: Vec<std::time::Instant>,
}

impl App<'_> {
    /// The latencies of the frames shown since the last look.
    fn note_shown(&mut self) {
        for (id, at) in self.host.take_presented() {
            // Frames are shown in order: one before this not shown yet was replaced, and never
            // will be (Metal drops a frame a newer one overtakes before its refresh).
            self.sampled.retain(|&k, _| k >= id);
            if let Some((sampled, presented)) = self.sampled.remove(&id) {
                self.latencies.push(at.saturating_duration_since(sampled).as_secs_f64() * 1e3);
                self.before_present
                    .push(presented.saturating_duration_since(sampled).as_secs_f64() * 1e3);
                self.shown.push(at);
            }
        }
    }
}

impl App<'_> {
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
        window.request_redraw();
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
            WindowEvent::RedrawRequested => {
                let (w, h) = self.size;
                // No more than `in_flight` frames between their input and their photons.
                let deadline = std::time::Instant::now() + std::time::Duration::from_millis(250);
                while self.sampled.len() >= self.run.in_flight.max(1) as usize
                    && std::time::Instant::now() < deadline
                {
                    std::thread::sleep(std::time::Duration::from_micros(250));
                    self.note_shown();
                }
                // The frame's input is what's come by now: its latency runs from here.
                let sampled = std::time::Instant::now();
                let drawn = self
                    .host
                    .lockstep_frame(self.frame, self.run.fps, w, h, &[])
                    .and_then(|()| self.host.present());
                match drawn {
                    Ok(Some(id)) => {
                        self.sampled.insert(id, (sampled, std::time::Instant::now()));
                    }
                    Ok(None) => {}
                    Err(e) => return self.fail(e, el),
                }
                self.note_shown();
                self.frame += 1;
                if self.run.frames.is_some_and(|n| self.frame >= n) {
                    el.exit();
                } else if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
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
    /// When each was shown.
    pub shown: Vec<std::time::Instant>,
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

    /// The same of input sampled to presented.
    pub fn before_present(&self) -> Option<(f64, f64, f64)> {
        stats(self.before_present.get(1..)?)
    }

    /// Frames shown a second after the first.
    pub fn rate(&self) -> Option<f64> {
        let (first, last) = (self.shown.get(1)?, self.shown.last()?);
        let span = last.saturating_duration_since(*first).as_secs_f64();
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
        sampled: std::collections::HashMap::new(),
        latencies: Vec::new(),
        before_present: Vec::new(),
        shown: Vec::new(),
    };
    event_loop.run_app(&mut app).map_err(|e| Error::Gpu(format!("the window: {e}")))?;
    if let Some(e) = app.failed {
        return Err(e);
    }
    // The last frames' photons come after the loop: what's shown within a moment counts.
    let until = std::time::Instant::now() + std::time::Duration::from_millis(250);
    while !app.sampled.is_empty() && std::time::Instant::now() < until {
        std::thread::sleep(std::time::Duration::from_millis(5));
        app.note_shown();
    }
    Ok(WindowRan {
        frames: app.frame,
        latencies: app.latencies,
        before_present: app.before_present,
        shown: app.shown,
    })
}
