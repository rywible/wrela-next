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
}

struct App<'a> {
    host: &'a mut Host,
    run: &'a WindowRun,
    window: Option<Arc<Window>>,
    size: (u32, u32),
    frame: u32,
    failed: Option<Error>,
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
                let drawn = self
                    .host
                    .lockstep_frame(self.frame, self.run.fps, w, h, &[])
                    .and_then(|()| self.host.present());
                if let Err(e) = drawn {
                    return self.fail(e, el);
                }
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

/// Runs `host`'s frames in a window as `run` says; the frames run.
pub fn run(host: &mut Host, run: &WindowRun) -> Result<u32> {
    let event_loop = EventLoop::new().map_err(|e| Error::Gpu(format!("no event loop: {e}")))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { host, run, window: None, size: run.size, frame: 0, failed: None };
    event_loop.run_app(&mut app).map_err(|e| Error::Gpu(format!("the window: {e}")))?;
    match app.failed {
        Some(e) => Err(e),
        None => Ok(app.frame),
    }
}
