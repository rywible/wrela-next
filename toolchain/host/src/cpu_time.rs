//! The CPU's side of a frame: where the host's own time goes, beside the GPU's spans.
//!
//! A frame's wall time on the host's thread splits four ways:
//!
//! - **check**: decoding, sequencing and checking each batch as the program submits it
//!   (`wrela.submit`, inside the program's call);
//! - **record**: decoding the call's batches again and carrying out each command (recording GPU
//!   work), and the end of the call (`Gpu::flush`, the submission);
//! - **blocked**: waiting for the GPU: a read-back, the guard's wait on the oldest frame, the
//!   serial mode's wait after each pass, reading timestamps;
//! - **program**: the rest, the program's own WASM (its frame and the ticks before it).
//!
//! Blocked time inside record is counted as blocked, not record. The counters are the
//! thread's: a host checks, records and waits on the thread that calls the program.

use std::cell::Cell;
use std::time::{Duration, Instant};

thread_local! {
    static CHECK: Cell<u64> = const { Cell::new(0) };
    static RECORD: Cell<u64> = const { Cell::new(0) };
    static BLOCKED: Cell<u64> = const { Cell::new(0) };
}

type Counter = std::thread::LocalKey<Cell<u64>>;

fn add(counter: &'static Counter, nanos: u64) {
    counter.with(|c| c.set(c.get() + nanos));
}

fn get(counter: &'static Counter) -> u64 {
    counter.with(Cell::get)
}

fn nanos(d: Duration) -> u64 {
    d.as_nanos() as u64
}

/// Runs `f`, counting its time as checking.
pub(crate) fn checking<T>(f: impl FnOnce() -> T) -> T {
    let began = Instant::now();
    let out = f();
    add(&CHECK, nanos(began.elapsed()));
    out
}

/// Runs `f`, counting its time as recording, less what it spent blocked.
pub(crate) fn recording<T>(f: impl FnOnce() -> T) -> T {
    let began = Instant::now();
    let blocked = get(&BLOCKED);
    let out = f();
    let waited = get(&BLOCKED).saturating_sub(blocked);
    add(&RECORD, nanos(began.elapsed()).saturating_sub(waited));
    out
}

/// Runs `f`, counting its time as blocked on the GPU.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn blocked<T>(f: impl FnOnce() -> T) -> T {
    let began = Instant::now();
    let out = f();
    add(&BLOCKED, nanos(began.elapsed()));
    out
}

/// One frame's CPU time on the host's thread, in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CpuFrame {
    pub frame: usize,
    /// The frame's wall time: its ticks, its call and everything the host did for them.
    pub total: f64,
    pub program: f64,
    pub check: f64,
    pub record: f64,
    pub blocked: f64,
}

/// Times one frame: the counters are zeroed, `f` runs, and the frame's split is read.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn frame<T>(frame: usize, f: impl FnOnce() -> T) -> (T, CpuFrame) {
    take();
    let began = Instant::now();
    let out = f();
    let total = nanos(began.elapsed());
    let [check, record, blocked] = take();
    let ms = |n: u64| n as f64 / 1e6;
    let program = total.saturating_sub(check + record + blocked);
    let split = CpuFrame {
        frame,
        total: ms(total),
        program: ms(program),
        check: ms(check),
        record: ms(record),
        blocked: ms(blocked),
    };
    (out, split)
}

/// The counters since the last take (check, record, blocked, in nanoseconds), zeroed.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
fn take() -> [u64; 3] {
    [&CHECK, &RECORD, &BLOCKED].map(|c| c.with(|c| c.replace(0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_time_inside_record_counts_as_blocked() {
        // The recording's own wall time: what record and blocked split between them, whatever
        // the sleeps took on a busy machine.
        let (wall, split) = frame(0, || {
            let began = Instant::now();
            recording(|| {
                std::thread::sleep(Duration::from_millis(4));
                blocked(|| std::thread::sleep(Duration::from_millis(10)));
            });
            let wall = began.elapsed().as_secs_f64() * 1e3;
            checking(|| std::thread::sleep(Duration::from_millis(2)));
            wall
        });
        assert!(split.blocked >= 10.0, "{split:?}");
        assert!(split.record >= 4.0, "{split:?}");
        // Counted once: as blocked, not as record too (that would be 10 ms more than the wall).
        assert!(split.record + split.blocked <= wall + 0.01, "{split:?} in {wall} ms");
        assert!(split.check >= 2.0, "{split:?}");
        let sum = split.program + split.check + split.record + split.blocked;
        assert!((sum - split.total).abs() < 0.01, "{split:?}");
    }
}
