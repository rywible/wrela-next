//! Play's schedule (language.md §16.2, test mode's two schedules: "Paced"): the ticker on a
//! thread and a clock of its own, `hz` ticks to a second of the wall clock, and the frames at the
//! display's pace, neither waiting for the other. A window runs a program so ([`crate::window`]):
//! in lockstep, test mode's other schedule, the sim's speed would be the display's (a 120 Hz
//! display's twice a 60 Hz one's).
//!
//! Tick `k` is due `origin + k / hz` seconds after the clock started (`std::tick::origin()`).
//! Each frame hands the ticker what it took ([`Ticker::queue`]: what the program posted, then
//! the frame's input), and the next tick to start takes it as its records, at most
//! `MAX_TICK_RECORDS`; the rest wait for the next. A script's records of a tick (`"tick": n`)
//! go after the queued ones, as in lockstep. A ticker [`BEHIND`] its clock or more (the machine
//! stalled) drops the ticks it couldn't run in time and moves the origin on by them, as the
//! browser host does; it never runs a dropped tick, so the ticks it runs are numbered on.
//!
//! What the ticker did comes back as [`Ran`]: each tick, when it took its records (under the
//! queue's lock, as [`Ticker::queue`] stamps when it queued), so a check can say each record
//! went to the first tick that started after it was queued ([`crate::window::check_ticks`]).

use crate::program::{TickThread, Ticked};
use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use wrela_abi::input::{EVENT_SIZE, Scripted, records_at};
use wrela_abi::memory::MAX_TICK_RECORDS;

/// How far behind its clock the ticker may fall before it drops ticks (s).
pub const BEHIND: f64 = 0.25;

type Record = [u8; EVENT_SIZE as usize];

/// What the ticker did.
pub enum Ran {
    /// A tick ran: what it gave, and when it took its records.
    Tick { ticked: Ticked, took: Instant },
    /// It dropped `ticks` ticks it couldn't run in time; tick 0 is now due at `origin`.
    Dropped { ticks: u32, origin: f64 },
    /// Tick `tick` trapped; the ticker stopped.
    Trapped { tick: u32, trap: wasmtime::Error },
}

struct Queue {
    records: VecDeque<Record>,
    stop: bool,
}

/// The ticker running on its own clock.
pub struct Ticker {
    thread: Option<JoinHandle<TickThread>>,
    queue: Arc<(Mutex<Queue>, Condvar)>,
    ran: mpsc::Receiver<Ran>,
    pub hz: u32,
    /// The program whose ticker it is ([`crate::Host::program_serial`]).
    pub(crate) program: u64,
}

impl Ticker {
    /// Starts `thread`'s ticks at `hz` on a clock started at `start` (frame time 0): its next
    /// tick due then. `script`'s records of ticks go to their ticks; `queued`, what the frames
    /// took before it started, to its first.
    pub(crate) fn start(
        thread: TickThread,
        hz: u32,
        program: u64,
        start: Instant,
        script: Vec<Scripted>,
        queued: Vec<Record>,
    ) -> Ticker {
        let queue = Arc::new((
            Mutex::new(Queue { records: queued.into_iter().collect(), stop: false }),
            Condvar::new(),
        ));
        let (send, ran) = mpsc::channel();
        let q = queue.clone();
        let thread = std::thread::Builder::new()
            .name("wrela ticker".into())
            .spawn(move || run(thread, hz, start, &script, &q, &send))
            .expect("a thread for the ticker");
        Ticker { thread: Some(thread), queue, ran, hz, program }
    }

    /// Queues `records` for the next tick to start: when they were queued.
    pub fn queue(&self, records: impl IntoIterator<Item = Record>) -> Instant {
        let mut q = lock(&self.queue.0);
        q.records.extend(records);
        Instant::now()
    }

    /// What the ticker did since the last look.
    pub fn take(&self) -> Vec<Ran> {
        self.ran.try_iter().collect()
    }

    /// Stops the ticker after the tick it's running: its instance back, and what it did since
    /// the last look.
    pub(crate) fn stop(mut self) -> (TickThread, Vec<Ran>) {
        lock(&self.queue.0).stop = true;
        self.queue.1.notify_all();
        let thread = self.thread.take().expect("joined once").join();
        let thread = thread.unwrap_or_else(|panic| std::panic::resume_unwind(panic));
        (thread, self.ran.try_iter().collect())
    }
}

fn lock(m: &Mutex<Queue>) -> std::sync::MutexGuard<'_, Queue> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The ticker's thread: each tick at its time, until stopped or a tick traps.
fn run(
    mut thread: TickThread,
    hz: u32,
    start: Instant,
    script: &[Scripted],
    queue: &(Mutex<Queue>, Condvar),
    ran: &mpsc::Sender<Ran>,
) -> TickThread {
    let period = 1.0 / f64::from(hz);
    let first = thread.next();
    // Tick `first` is due at the start: tick 0 was due `first` ticks before.
    let mut origin = -f64::from(first) * period;
    thread.set_origin(origin);
    loop {
        let k = thread.next();
        let due = start + Duration::from_secs_f64((origin + f64::from(k) * period).max(0.0));
        let (records, took) = {
            let mut q = lock(&queue.0);
            loop {
                if q.stop {
                    return thread;
                }
                let now = Instant::now();
                if now >= due {
                    break;
                }
                q = match queue.1.wait_timeout(q, due - now) {
                    Ok((q, _)) => q,
                    Err(poisoned) => poisoned.into_inner().0,
                };
            }
            let late = Instant::now().saturating_duration_since(due).as_secs_f64();
            if late >= BEHIND {
                // Ticks it couldn't run in time: dropped, and the clock moved on by them, so
                // this tick is due now.
                let ticks = (late / period).floor() as u32;
                origin += f64::from(ticks) * period;
                thread.set_origin(origin);
                let _ = ran.send(Ran::Dropped { ticks, origin });
            }
            q.records.extend(records_at(script, k).map(|e| e.bytes()));
            let n = q.records.len().min(MAX_TICK_RECORDS as usize);
            (q.records.drain(..n).collect::<Vec<_>>(), Instant::now())
        };
        match thread.run(records) {
            Ok(ticked) => {
                if ran.send(Ran::Tick { ticked, took }).is_err() {
                    return thread;
                }
            }
            Err((tick, trap)) => {
                let _ = ran.send(Ran::Trapped { tick, trap });
                return thread;
            }
        }
    }
}
