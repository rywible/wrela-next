//! The program's threads (#43 §2–§3, AC6): tasks (`std::par::spawn`), parallel work started
//! inside parallel work, hand-offs (`std::handoff`) and per-thread allocation counts, natively
//! with 1, 2 and 8 threads (the program's own and its helpers). toolchain/tests/jobs is the
//! program.

use crate::{a_flipped_record_fails_at_its_tick, built, replays_with_any_helpers};
use wrela_host::{CpuBuild, CpuHost, Value};
use wrela_tests::{one_u32, one_vec3};

fn start(workers: u32) -> CpuHost {
    CpuBuild::load(built("jobs")).expect("load").start_with(workers).expect("start")
}

/// A job's result is a function of its input alone, whichever thread runs it: twelve jobs at
/// once, and a job that starts and joins another, give the same with no helpers (every job runs
/// in its `join`), one, and seven. With one helper the inner job runs inside the outer's `join`,
/// so a job that joins another can't deadlock.
#[test]
fn jobs_give_the_same_results_with_any_helpers() {
    let results = |workers| {
        let mut host = start(workers);
        let many = one_u32(&mut host, "many", &[Value::I32(12), Value::I32(200_000)]);
        let nested = one_u32(&mut host, "nested_jobs", &[Value::I32(300_000)]);
        (many, nested)
    };
    let one = results(1);
    for workers in [2, 8] {
        assert_eq!(results(workers), one, "with {workers} threads");
    }
}

/// A job a program polls (`done`) is done before long, with no helpers too: then the first poll
/// runs it, since no helper would ever start it (a page on a two-core machine has none; a program
/// that polled its preview waited forever in test mode's one thread).
#[test]
fn a_polled_job_is_done_with_any_helpers() {
    const MOST: u32 = 2_000_000_000;
    let mut host = start(1);
    assert_eq!(one_u32(&mut host, "polled", &[Value::I32(200_000), Value::I32(MOST as i32)]), 0);
    for workers in [2, 8] {
        let mut host = start(workers);
        let polls = one_u32(&mut host, "polled", &[Value::I32(200_000), Value::I32(MOST as i32)]);
        assert!(polls < MOST, "{workers} threads: the job was never done");
    }
}

/// A trap in a job is the program's panic at its `join`, with the job's message, whichever
/// thread ran it: with no helpers, one or seven, and with each job's result held back (slowed)
/// so the joining thread waits for it.
#[test]
fn a_trapping_job_traps_at_its_join() {
    for (workers, hold) in [(1, 0), (2, 0), (8, 0), (2, 50_000), (8, 50_000)] {
        let mut host = start(workers);
        host.hold_jobs(hold);
        assert_eq!(one_u32(&mut host, "trap_at_join", &[Value::I32(4)]), 8);
        let e = host.call_export("trap_at_join", &[Value::I32(13)]).expect_err("13 traps");
        let msg = e.to_string();
        assert!(msg.contains("a job's input was 13"), "{workers} threads, held {hold}: {msg}");
        let failure = host.last_failure().expect("a trap");
        assert_eq!(failure.panic.as_deref(), Some("a job's input was 13"));
    }
}

#[test]
fn a_trapping_job_traps_when_dropped() {
    for (workers, hold) in [(1, 0), (2, 0), (8, 0), (2, 50_000), (8, 50_000)] {
        let mut host = start(workers);
        host.hold_jobs(hold);
        assert_eq!(one_u32(&mut host, "trap_at_drop", &[Value::I32(4)]), 8);
        let error = host.call_export("trap_at_drop", &[Value::I32(13)]).expect_err("13 traps");
        assert!(error.to_string().contains("a job's input was 13"));
        assert_eq!(
            host.last_failure().expect("a trap").panic.as_deref(),
            Some("a job's input was 13")
        );
    }
}

/// Shrink only the fixture's imported memory maximum, so memory.grow fails under the real
/// allocator lock after a few blocks. No production allocator or scheduling path is replaced.
fn small_heap() -> std::path::PathBuf {
    use wasm_encoder::reencode::Reencode;
    struct Heap;
    impl Reencode for Heap {
        type Error = std::convert::Infallible;
        fn memory_type(
            &mut self,
            ty: wasmparser::MemoryType,
        ) -> Result<wasm_encoder::MemoryType, wasm_encoder::reencode::Error<Self::Error>> {
            let mut ty = wasm_encoder::reencode::utils::memory_type(self, ty);
            ty.maximum = Some(ty.minimum + 64);
            Ok(ty)
        }
    }
    let dir = crate::scratch("jobs-small-heap");
    wrela_tests::copy_dir(&built("jobs"), &dir);
    let manifest = wrela_tests::manifest(&dir);
    let wasm = dir.join(&manifest.wasm);
    let bytes = std::fs::read(&wasm).unwrap();
    let mut module = wasm_encoder::Module::new();
    Heap.parse_core_module(&mut module, wasmparser::Parser::new(0), &bytes).unwrap();
    std::fs::write(wasm, module.finish()).unwrap();
    dir
}

/// OOM is reported at join, after releasing the allocator lock, and another thread can
/// continue allocating. One forced failure with 1, 2 and 8 workers replaces repeated GiB fills.
#[test]
fn an_allocation_failure_releases_the_lock_and_panics_at_join() {
    let build = CpuBuild::load(small_heap()).expect("load");
    for workers in [1, 2, 8] {
        let mut host = build.start_with(workers).expect("start");
        let error = host.call_export("fill_the_heap_while_allocating", &[]).expect_err("OOM");
        let failure = host.last_failure().expect("a trap");
        assert!(
            failure.panic.as_deref().is_some_and(|m| m.starts_with("out of memory")),
            "{error}"
        );
        // If the failing thread retained the lock, this allocation would deadlock.
        assert!(one_u32(&mut host, "allocations_while_helping", &[Value::I32(1)]) > 0);
    }
}

/// Parallel work started inside a parallel closure runs on that thread alone, chunk after
/// chunk: the results are the same with any helpers, and nothing deadlocks.
#[test]
fn parallel_work_inside_parallel_work_runs_inline() {
    let run = |workers| one_u32(&mut start(workers), "nested_parallel", &[Value::I32(64)]);
    let one = run(1);
    assert_eq!(run(2), one);
    assert_eq!(run(8), one);
}

/// Each thread counts its own allocations: what this thread counts making jobs is the same
/// whether helpers run them (and allocate) or it does.
#[test]
fn allocations_are_counted_per_thread() {
    let run =
        |workers| one_u32(&mut start(workers), "allocations_while_helping", &[Value::I32(16)]);
    let alone = run(1);
    assert!(alone > 0);
    assert_eq!(run(8), alone);
}

/// A hand-off is never read torn (AC6): a helper publishes as fast as it can while this thread
/// reads, in a debug build, where each read checks its value's checksum and that it's no older
/// than the last; and the program checks each pair it reads, and that it's the latest complete
/// (no older than one the helper had finished publishing before the read). 0 mismatches in 10⁶
/// reads.
#[test]
fn a_hand_off_is_never_read_torn() {
    let dir = crate::built_debug("jobs");
    let mut host = CpuBuild::load(&dir).expect("load").start_with(2).expect("start");
    let (values, reads) = (20_000, 100_000);
    let r = host
        .call_export("stress", &[Value::I32(values), Value::I32(reads)])
        .unwrap_or_else(|e| panic!("stress: {e}"));
    let [Value::F32(count), Value::F32(published)] = r[..] else { panic!("{r:?}") };
    println!("{count} reads while {values} values were published; the last read saw {published}");
    assert!(count >= reads as f32);
    assert_eq!(published, values as f32);
}

fn ticker(workers: u32) -> CpuHost {
    CpuBuild::load(built("ticker")).expect("load").start_with(workers).expect("start")
}

/// A ticker (`std::tick`) in lockstep with the frames (#43 §2.3): before frame i, a 60 Hz
/// ticker has run i + 1 ticks at 60 frames a second, so frame i reads tick i's snapshot through
/// the hand-off; a script's tick-keyed events reach their ticks as records, and its
/// frame-keyed ones reach both the frames and the next tick.
#[test]
fn a_ticker_takes_its_records_and_publishes_to_the_frames() {
    let script = wrela_host::parse_script(
        r#"[
            {"tick": 3, "type": "key", "key": "Space"},
            {"frame": 5, "type": "keydown", "key": "KeyA"},
            {"tick": 20, "type": "keydown", "key": "KeyB"}
        ]"#,
    )
    .expect("a script");
    let mut host = ticker(2);
    assert_eq!(host.ticker_hz(), Some(60));
    for i in 0..30 {
        host.lockstep_frame(i, 60.0, 64, 64, &script).expect("a frame");
    }
    let [tick, keys, _] = one_vec3(&mut host, "newest", &[]);
    assert_eq!((tick, keys), (29.0, 3.0));
    let e = host.call_export("start_again", &[]).expect_err("a second ticker");
    assert!(e.to_string().contains("a program starts one ticker"), "{e}");
}

/// Parallel work started from the ticker's thread and from the frames' at the same time gives
/// the same results as each alone (AC6): ticks on an OS thread of their own beside frames, with
/// seven helpers, give the same state hashes as ticks alone, and the frames the same work.
#[test]
fn parallel_work_from_ticks_and_frames_doesnt_collide() {
    let (ticks, frames) = (60, 60);
    let built = CpuBuild::load(built("ticker")).expect("load");
    let alone = built.record_ticks(ticks, &[], 1).expect("ticks alone");
    let alone: Vec<u64> = alone.ticks.iter().map(|t| t.hash).collect();
    let mut frames_alone = built.start_with(1).expect("start");
    for i in 0..frames {
        frames_alone.frame(wrela_host::frame_time(i, 60.0), 64, 64).expect("a frame");
    }
    let work = one_u32(&mut frames_alone, "work", &[]);
    for round in 0..2 {
        let mut host = built.start_with(8).expect("start");
        let ticked = host.ticks_beside_frames(ticks, frames, 60.0, 64, 64).expect("both");
        let hashes: Vec<u64> = ticked.iter().map(|(t, _)| t.hash.expect("asked for")).collect();
        assert_eq!(hashes, alone, "round {round}: the ticks' hashes");
        assert_eq!(one_u32(&mut host, "work", &[]), work, "round {round}");
    }
}

/// A tick log (runtime/abi `ticks`) replays to the same hashes (`wrela-host --replay`), with
/// any helpers, and a log with one record changed fails at that tick and names it (AC6).
#[test]
fn a_tick_log_replays_and_a_changed_record_fails_at_its_tick() {
    let script = wrela_host::parse_script(
        r#"[
            {"tick": 3, "type": "key", "key": "Space"},
            {"tick": 10, "type": "keydown", "key": "KeyA"},
            {"tick": 40, "type": "keyup", "key": "KeyA"}
        ]"#,
    )
    .expect("a script");
    let built = CpuBuild::load(built("ticker")).expect("load");
    let log = built.record_ticks(60, &script, 1).expect("record");
    let read = replays_with_any_helpers(&built, &log);
    // A key going down at tick 10 becomes one going up: tick 10's hash differs.
    a_flipped_record_fails_at_its_tick(&built, &read, 10);
    let mut other = read.clone();
    other.wasm_hash ^= 1;
    assert!(matches!(built.replay(&other, 1), Err(wrela_host::ReplayError::OtherBuild { .. })));
}

/// The posted test program's script: a key down before frame 4, which reaches tick 4 after
/// what frame 3 posted.
const POSTED_SCRIPT: &str = r#"[{"frame": 4, "type": "keydown", "key": "KeyA"}, {"frame": 11, "type": "keydown", "key": "KeyB"}]"#;

/// Records a program posts into its ticks (`std::tick::post`): in lockstep, what frame `i`
/// posts is a record of the first tick that runs for frame `i + 1`, before that frame's input,
/// in the order posted, and what `init` posts once the ticker has started is tick 0's; past a
/// tick's records, the rest wait in order, ahead of the input after them; the tick log keeps
/// them, so the log replays with any helpers (a replay drops what the program posts again); a
/// changed post fails at its tick; and the frames see the sim that took them.
#[test]
fn posted_records_are_the_next_ticks_and_replay() {
    use wrela_abi::input::Event;
    let script = wrela_host::parse_script(POSTED_SCRIPT).expect("a script");
    let built = CpuBuild::load(built("posted")).expect("load");
    let frames = 14;
    let log = built.record_lockstep(frames, 60.0, &script, 1).expect("record");
    assert_eq!(log.ticks.len() as u32, wrela_host::lockstep_ticks(frames - 1, 60, 60.0));
    let key = |k| Event::key(true, wrela_abi::input::key_code(k), false, 0).bytes();
    let records = |k: usize| log.ticks[k].records.clone();
    assert_eq!(records(0), vec![Event::posted(2, 42, 0).bytes()], "tick 0: init's post");
    assert_eq!(records(4), vec![Event::posted(1, 5, 0).bytes(), key("KeyA")], "tick 4");
    assert_eq!(
        records(8),
        vec![Event::posted(2, u64::MAX, 1 << 32).bytes(), Event::posted(1, 7, 0).bytes()],
        "tick 8"
    );
    let note = |i: u64| Event::posted(2, i, 0).bytes();
    let max = wrela_abi::memory::MAX_TICK_RECORDS as u64;
    assert_eq!(records(11), (0..max).map(note).collect::<Vec<_>>(), "tick 11: a tick's worth");
    let rest: Vec<_> = (max..300).map(note).chain([key("KeyB")]).collect();
    assert_eq!(records(12), rest, "tick 12: the rest of the posts, then frame 11's input");
    let others = (0..log.ticks.len()).filter(|k| ![0, 4, 8, 11, 12].contains(k));
    assert!(others.into_iter().all(|k| records(k).is_empty()), "no other tick has records");
    let read = replays_with_any_helpers(&built, &log);
    // Frame 7's second post, levels 0 to 2, made levels 1 and 2: tick 8 differs.
    let mut changed = read.clone();
    changed.ticks[8].records[1] = Event::posted(1, 6, 0).bytes();
    match built.replay(&changed, 2) {
        Err(wrela_host::ReplayError::Tick { tick, .. }) => assert_eq!(tick, 8),
        other => panic!("expected tick 8 to differ: {other:?}"),
    }
    let mut host = built.start_with(2).expect("start");
    for i in 0..frames {
        host.lockstep_frame(i, 60.0, 64, 64, &script).expect("a frame");
    }
    let [tick, opened, posts] = one_vec3(&mut host, "newest", &[]);
    assert_eq!((tick, opened, posts), (13.0, 7.0, 304.0));
    assert_eq!(one_u32(&mut host, "keys", &[]), 2);
}

/// A program that posts into its ticks with no ticker fails its frame's commands.
#[test]
fn a_post_without_a_ticker_fails() {
    let built = CpuBuild::load(built("posted-unticked")).expect("load");
    let mut host = built.start_with(1).expect("start");
    let e = host.frame(0.0, 64, 64).expect_err("no ticker");
    assert!(e.to_string().contains("has started no ticker"), "{e}");
}

/// Memory grown by one thread is usable by all helpers. A 4 MiB initialization overlaps helper
/// startup; one helper holds 12 MiB while parallel rows write beyond it, then the main thread
/// writes every page of 16 MiB, and reports what it used.
#[test]
#[ignore = "needs the runtime"]
fn memory_a_helper_grew_is_every_threads() {
    let (dir, _) = wrela_tests::page("toolchain/tests/grown", "grown");
    let native = {
        let mut host = wrela_host::Host::load(&dir).expect("load");
        let (mut lines, began) = (Vec::new(), std::time::Instant::now());
        for i in 0.. {
            assert!(began.elapsed().as_secs() < 60, "natively, the helper never finished");
            host.frame(i as f32 / 60.0, 16, 16).expect("a frame");
            lines.extend(host.take_logs());
            if lines.iter().any(|l| l.starts_with("used")) {
                break;
            }
        }
        lines.into_iter().find(|l| l.starts_with("used")).expect("natively, the block's used")
    };
    assert!(!native.is_empty());
}
