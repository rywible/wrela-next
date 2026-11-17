//! Frame pacing: when a windowed frame takes its input, so that it reaches the display at the
//! refresh it aims for with as little time as it can between its input and its photons.
//!
//! The pacer learns the display's refreshes from when frames are shown (their period and their
//! phase), how long a frame takes from its input to its present (`work`, the slowest of the
//! recent ones but a few), and how long a present needs before the refresh that shows it
//! (`latch`: the GPU's part, the compositor's, the display's). A frame aims at one refresh, one
//! frame a refresh, and takes its input at that refresh less work and latch. A frame shown later
//! than its refresh is a miss: the latch grows past what that frame needed, and the pacer waits
//! longer each time before trying a shorter one, so misses get rarer the longer a run goes.
//!
//! Times are seconds on one clock. Pure arithmetic: the window drives it, and the tests drive it
//! against a display simulated on the CPU.

use std::collections::{BTreeMap, VecDeque};

/// When the next frame takes its input, and the refresh it aims to be shown at (`None` before
/// any frame has been shown: then frames go as they're done).
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub at: f64,
    pub target: Option<f64>,
}

pub struct Pacer {
    /// The display's refresh period, as frames shown say.
    period: f64,
    gaps: VecDeque<f64>,
    /// A refresh: when the last frame was shown.
    grid: Option<f64>,
    /// Recent frames' input to present.
    work: VecDeque<f64>,
    /// Present to the refresh that shows it, at the least.
    latch: f64,
    /// Frames presented and not yet shown, by id: the refresh each aimed at, and its present.
    waiting: BTreeMap<u64, (Option<f64>, f64)>,
    last_target: Option<f64>,
    /// Frames shown on time since the latch last moved, and how many it takes to try a shorter
    /// latch (after the first miss; before it, the latch comes down fast).
    hits: u32,
    patience: u32,
    missed_ever: bool,
    /// Frames shown after their refresh, or never.
    pub misses: u32,
    /// Frames shown so far: the first [`LEARN`] go unpaced, and say the refresh period.
    shown: u32,
}

/// Frames shown unpaced at the start, as fast as the display takes them: their gaps are its
/// refresh period (a paced frame aims at every refresh it thinks there is, so it can't see one
/// faster).
const LEARN: u32 = 16;

/// How far after its time a frame may still take its input and keep its refresh: a late wake.
const LATE: f64 = 0.0005;
/// Past what a missed frame needed, the latch's margin.
const MARGIN: f64 = 0.001;
/// A frame shown more than this after its aim was late.
const SHOWN_LATE: f64 = 0.002;
/// A frame on time with no more than this past the latch to spare tested the latch.
const TIGHT: f64 = 0.001;
/// Paced frames shown before a miss counts as the display's.
const SETTLE: u32 = 8;
/// A frame slower than this many refreshes from its input to its present is a hitch.
const HITCH: f64 = 4.0;
/// The furthest ahead a plan may put a frame's input: whatever the pacer thinks, a window never
/// waits longer than this for its next frame.
const LONGEST_WAIT: f64 = 0.1;

impl Pacer {
    /// A pacer for a display thought to refresh every `period` seconds until frames shown say.
    pub fn new(period: f64) -> Pacer {
        Pacer {
            period,
            gaps: VecDeque::new(),
            grid: None,
            work: VecDeque::new(),
            latch: period,
            waiting: BTreeMap::new(),
            last_target: None,
            hits: 0,
            patience: 60,
            missed_ever: false,
            misses: 0,
            shown: 0,
        }
    }

    /// The display's refresh period as learned (seconds).
    pub fn period(&self) -> f64 {
        self.period
    }

    /// Input to photons, as the pacer now plans it (seconds).
    pub fn budget(&self) -> f64 {
        self.work_estimate() + self.latch
    }

    /// The pacer's state in a line, for `WRELA_PACE_TRACE`.
    pub fn describe(&self) -> String {
        format!(
            "period {:.2} ms, latch {:.2} ms, work {:.2} ms, waiting {:?}, last aim {:?}, misses {}",
            self.period * 1e3,
            self.latch * 1e3,
            self.work_estimate() * 1e3,
            self.waiting
                .iter()
                .map(|(id, (aim, p))| (
                    *id,
                    aim.map(|a| (a * 1e4).round() / 10.0),
                    (p * 1e4).round() / 10.0
                ))
                .collect::<Vec<_>>(),
            self.last_target.map(|a| (a * 1e4).round() / 10.0),
            self.misses
        )
    }

    /// When the next frame should take its input, at `now`.
    pub fn plan(&self, now: f64) -> Plan {
        let Some(grid) = self.grid.filter(|_| self.shown >= LEARN) else {
            // Refreshes not known yet: frames go as they're done, two waiting at the most.
            let at = if self.waiting.len() >= 2 && self.waiting_since(now) < LONGEST_WAIT {
                now + 0.003
            } else {
                now
            };
            return Plan { at, target: None };
        };
        let t = self.period;
        let need = self.budget();
        // Frames that never come back (a window hidden) don't pile up.
        if self.waiting.len() as f64 > (need / t).ceil() + 2.0
            && self.waiting_since(now) < LONGEST_WAIT
        {
            return Plan { at: now + 0.003, target: None };
        }
        // The first refresh that input taken now can still make, and after the last aimed at:
        // one frame a refresh.
        let mut k = ((now - LATE + need - grid) / t).ceil();
        if let Some(last) = self.last_target {
            k = k.max(((last - grid) / t).round() + 1.0);
        }
        let target = grid + k * t;
        Plan { at: (target - need).min(now + LONGEST_WAIT), target: Some(target) }
    }

    /// Frame `id` took its input at `sampled` and was presented at `presented`, aiming at
    /// `target`.
    pub fn presented(&mut self, id: u64, sampled: f64, presented: f64, target: Option<f64>) {
        // Not a hitch (pipelines built, say): no plan can pace a frame slower than a few
        // refreshes, and one would hold every plan back for 120 frames.
        let work = presented - sampled;
        if work < HITCH * self.period.min(1.0 / 30.0) {
            push(&mut self.work, work, 120);
        }
        self.waiting.insert(id, (target, presented));
        if target.is_some() {
            self.last_target = target;
        }
    }

    /// Frame `id` was shown at `at`. Frames are shown in order: one before it still waiting was
    /// overtaken, and won't be.
    pub fn shown(&mut self, id: u64, at: f64) {
        let overtaken: Vec<u64> = self.waiting.range(..id).map(|(&k, _)| k).collect();
        for k in overtaken {
            if let Some((Some(target), presented)) = self.waiting.remove(&k) {
                self.miss(target - presented);
            }
        }
        let Some((target, presented)) = self.waiting.remove(&id) else { return };
        // Paced, the gaps between frames shown are the pacer's own doing (where a present
        // isn't held to a refresh, each a little late would teach it a slower one, and slower):
        // only the unpaced start says the display's period.
        if let (Some(last), true) = (self.grid, self.shown < LEARN) {
            let gap = at - last;
            if gap > 0.0 {
                push(&mut self.gaps, gap, 64);
                self.period = quantile(&self.gaps, 0.5);
            }
        }
        self.shown += 1;
        let Some(target) = target else {
            self.grid = Some(at);
            return;
        };
        let t = self.period;
        // Each frame shown says where the refreshes are: exactly, on a display that holds
        // frames to its refresh (a lattice of our own drifts with any error in the period).
        self.grid = Some(at);
        // Late: past its refresh, or, where a display shows a frame whenever it takes it, more
        // than a hair past its aim (each such frame would push every later one later).
        if at > target + SHOWN_LATE {
            // It needed more than it had between its present and its refresh.
            self.miss(target - presented);
            // The frames still waiting were aimed at refreshes it took: they go a refresh
            // each after it, and the next frame after them.
            let mut next = at + t;
            for (aim, _) in self.waiting.values_mut() {
                if let Some(aim) = aim {
                    *aim = aim.max(next);
                    next = *aim + t;
                }
            }
            self.last_target = Some(self.last_target.map_or(next - t, |l| l.max(next - t)));
        } else if target - presented < self.latch + TIGHT {
            // On time with no more than the latch to spare: evidence the latch is enough. (A
            // frame whose work came in under the estimate had more, and says nothing of it.)
            self.hits += 1;
            let (wait, step) = if self.missed_ever { (self.patience, 0.0001) } else { (1, 0.0005) };
            if self.hits >= wait {
                self.hits = 0;
                self.latch = (self.latch - step).max(0.0);
            }
        }
    }

    /// A frame missed its refresh with `had` between its present and that refresh.
    fn miss(&mut self, had: f64) {
        self.misses += 1;
        self.hits = 0;
        // Its work overran the estimate, leaving less than the latch: the work estimate has it
        // now, and the latch wasn't at fault.
        if had < self.latch - 0.0002 {
            return;
        }
        // The first paced frames may meet frames still queued from the unpaced start: a miss
        // then says nothing of the display, and the latch keeps coming down fast.
        if self.shown < LEARN + SETTLE {
            self.latch += MARGIN;
            return;
        }
        if self.missed_ever {
            self.patience = (self.patience * 2).min(7200);
        }
        self.missed_ever = true;
        // At most two margins a miss: a frame that missed with time to spare past the latch
        // met a hiccup (the compositor late once) more likely than a display that needs that
        // much more, and one such would hold the latch up for the rest of the run. A display that
        // does need more misses its way up, two margins a time.
        self.latch = (had + MARGIN).min(self.latch + 2.0 * MARGIN);
    }

    /// How long the oldest frame not yet shown has waited since its present.
    fn waiting_since(&self, now: f64) -> f64 {
        self.waiting.values().next().map_or(0.0, |&(_, presented)| now - presented)
    }

    /// Input to present, as planned: the slowest of the last 120 frames, hitches aside (a slow
    /// frame that comes back is planned for rather than missed).
    fn work_estimate(&self) -> f64 {
        self.work.iter().copied().fold(0.0, f64::max)
    }
}

fn push(window: &mut VecDeque<f64>, v: f64, most: usize) {
    if window.len() == most {
        window.pop_front();
    }
    window.push_back(v);
}

fn quantile(window: &VecDeque<f64>, q: f64) -> f64 {
    let mut v: Vec<f64> = window.iter().copied().collect();
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A display simulated on the CPU: it refreshes every `period` from `phase`, and shows a
    /// frame at the first refresh at least `latch` after its present, one frame a refresh in
    /// order (FIFO); or, from frame `gridless_from`, `latch` after its present whenever that is
    /// (a compositor that says a frame is shown when it takes it). Each frame's input to present
    /// is `work(frame)`.
    struct Display {
        period: f64,
        phase: f64,
        latch: f64,
        gridless_from: u64,
        /// A frame the display shows a refresh late whatever it had (a compositor hiccup).
        hiccup: Option<u64>,
    }

    struct Ran {
        latencies: Vec<f64>,
        shown: Vec<f64>,
        misses: u32,
    }

    fn run(display: &Display, frames: u64, guess: f64, work: impl Fn(u64) -> f64) -> Ran {
        let mut pacer = Pacer::new(guess);
        let (mut now, mut last_shown) = (0.0f64, f64::NEG_INFINITY);
        // Shown, not yet told: (when, id, sampled).
        let mut coming: Vec<(f64, u64, f64)> = Vec::new();
        let mut ran = Ran { latencies: Vec::new(), shown: Vec::new(), misses: 0 };
        for id in 0..frames {
            // As the window does: take what's been shown, plan, and if the plan is more than
            // 2 ms off, wake 1 ms before it and plan again.
            let plan = loop {
                coming.sort_by(|a, b| a.0.total_cmp(&b.0));
                for &(at, id, sampled) in coming.iter().filter(|c| c.0 <= now) {
                    pacer.shown(id, at);
                    ran.latencies.push(at - sampled);
                    ran.shown.push(at);
                }
                coming.retain(|c| c.0 > now);
                let plan = pacer.plan(now);
                if plan.at - now > 0.002 {
                    now = plan.at - 0.001;
                    continue;
                }
                break plan;
            };
            let sampled = plan.at.max(now);
            if std::env::var_os("PACE_SIM_TRACE").is_some() && id % 50 == 0 {
                eprintln!("{id}: {:?} {}", plan, pacer.describe());
            }
            let presented = sampled + work(id);
            pacer.presented(id, sampled, presented, plan.target);
            let k = ((presented + display.latch - display.phase) / display.period).ceil();
            let mut at = display.phase + k * display.period;
            if id >= display.gridless_from {
                // Taken a little after its present, whenever: late by a varying 0 to 0.5 ms.
                at = presented + display.latch + (id % 5) as f64 * 0.0001;
            } else if at <= last_shown {
                at = last_shown + display.period;
            }
            if display.hiccup == Some(id) {
                at += display.period;
            }
            last_shown = at;
            coming.push((at, id, sampled));
            now = presented;
        }
        coming.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (at, id, sampled) in coming {
            pacer.shown(id, at);
            ran.latencies.push(at - sampled);
            ran.shown.push(at);
        }
        ran.misses = pacer.misses;
        ran
    }

    fn median(v: &[f64]) -> f64 {
        quantile(&v.iter().copied().collect(), 0.5)
    }

    #[test]
    fn a_paced_frame_takes_its_input_just_in_time_at_every_refresh() {
        let display = Display {
            period: 1.0 / 60.0,
            phase: 0.0031,
            latch: 0.004,
            gridless_from: u64::MAX,
            hiccup: None,
        };
        // 3 to 5 ms of work, a 9 ms frame now and then.
        let work = |i: u64| if i % 97 == 13 { 0.009 } else { 0.003 + (i % 7) as f64 * 0.0003 };
        // Only the 9 ms frames test the latch (the others have time to spare): it comes down
        // one of those at a time.
        let ran = run(&display, 8000, 1.0 / 60.0, work);
        let settled = &ran.latencies[4000..];
        let m = median(settled);
        // Input to photons: the slowest frame's work, the display's latch and a margin (about
        // 14 ms, the 9 ms frames absorbed rather than missed), far under two refreshes' 33.
        assert!(m < 0.0145, "median {m}");
        // Steady: the worst settled frame within 2 ms of the median.
        let worst = settled.iter().copied().fold(0.0, f64::max);
        assert!(worst < m + 0.002, "worst {worst} against {m}");
        // One frame each refresh once settled.
        let span = ran.shown[7999] - ran.shown[4000];
        let refreshes = (span / display.period).round() as usize;
        assert!(refreshes <= 3999 + 2, "{refreshes} refreshes for 3999 frames");
        // Misses are few, and early: the latch climbs to the display's a millisecond a miss.
        assert!(ran.misses <= 8, "{} misses", ran.misses);
    }

    #[test]
    fn the_pacer_learns_a_refresh_it_was_not_told() {
        // A 120 Hz display the pacer thinks is 60 Hz.
        let display = Display {
            period: 1.0 / 120.0,
            phase: 0.0,
            latch: 0.002,
            gridless_from: u64::MAX,
            hiccup: None,
        };
        let ran = run(&display, 900, 1.0 / 60.0, |_| 0.002);
        let gaps: Vec<f64> = ran.shown[600..].windows(2).map(|w| w[1] - w[0]).collect();
        let g = median(&gaps);
        assert!((g - 1.0 / 120.0).abs() < 0.0002, "shown every {g}");
        assert!(median(&ran.latencies[600..]) < 0.0075);
    }

    #[test]
    fn a_present_slower_than_a_refresh_still_goes_one_a_refresh_with_two_in_flight() {
        // 12 ms of work, and the GPU and compositor need 14 more: more than a refresh.
        let display = Display {
            period: 1.0 / 60.0,
            phase: 0.0,
            latch: 0.014,
            gridless_from: u64::MAX,
            hiccup: None,
        };
        let ran = run(&display, 600, 1.0 / 60.0, |_| 0.012);
        let settled = &ran.latencies[300..];
        // 26 ms or so, and every refresh is still used.
        assert!(median(settled) < 0.0305, "median {}", median(settled));
        let gaps: Vec<f64> = ran.shown[300..].windows(2).map(|w| w[1] - w[0]).collect();
        assert!((median(&gaps) - display.period).abs() < 0.0002);
    }

    #[test]
    fn a_hitch_does_not_hold_back_the_frames_after_it() {
        // Pipelines built in the second frame and again in the 400th: 3 s each.
        let display = Display {
            period: 1.0 / 60.0,
            phase: 0.002,
            latch: 0.004,
            gridless_from: u64::MAX,
            hiccup: None,
        };
        let ran = run(&display, 900, 1.0 / 60.0, |i| if i == 1 || i == 400 { 3.0 } else { 0.004 });
        let settled = &ran.latencies[600..];
        assert!(median(settled) < 0.0115, "median {}", median(settled));
        let gaps: Vec<f64> = ran.shown[600..].windows(2).map(|w| w[1] - w[0]).collect();
        assert!((median(&gaps) - display.period).abs() < 0.0002);
    }

    #[test]
    fn frames_never_shown_hold_the_next_back_no_longer_than_a_tenth_of_a_second() {
        let t = 1.0 / 60.0;
        let mut pacer = Pacer::new(t);
        let mut now = 0.0;
        for id in 0..200u64 {
            let plan = pacer.plan(now);
            assert!(plan.at - now <= LONGEST_WAIT + 1e-9, "frame {id} waits {}", plan.at - now);
            now = plan.at.max(now);
            pacer.presented(id, now, now + 0.004, plan.target);
            now += 0.004;
            // Shown on time for a while, then never again (the window hidden, say).
            if id < 60 {
                pacer.shown(id, plan.target.unwrap_or(((now + 0.004) / t).ceil() * t));
            }
        }
    }

    #[test]
    fn a_display_that_shows_frames_off_any_grid_does_not_slow_the_pacer_down() {
        // Headless Weston under mailbox, say: 40 Hz while unpaced, and then whenever.
        let display =
            Display { period: 0.025, phase: 0.0, latch: 0.004, gridless_from: 40, hiccup: None };
        let ran = run(&display, 3000, 1.0 / 60.0, |_| 0.004);
        let gaps: Vec<f64> = ran.shown[2000..].windows(2).map(|w| w[1] - w[0]).collect();
        let g = median(&gaps);
        // Each frame may be shown up to SHOWN_LATE after its aim before it counts as late: no
        // slower than that a frame, and no runaway.
        assert!(g >= 0.025 - 0.0005 && g <= 0.025 + SHOWN_LATE, "shown every {g}");
    }

    #[test]
    fn a_compositor_that_takes_a_refresh_costs_one_refresh_and_no_more() {
        // The M4 in a window: a present is shown a refresh after the compositor takes it, so
        // about 17 ms after it at the least; 4 to 5 ms of work.
        let display = Display {
            period: 1.0 / 60.0,
            phase: 0.0041,
            latch: 0.0172,
            gridless_from: u64::MAX,
            hiccup: None,
        };
        let ran = run(&display, 1500, 1.0 / 60.0, |i| 0.0042 + (i % 5) as f64 * 0.0002);
        let settled = &ran.latencies[750..];
        let m = median(settled);
        assert!(m < 0.0245, "median {m}, {} misses", ran.misses);
    }

    #[test]
    fn a_compositor_hiccup_on_a_frame_with_time_to_spare_does_not_hold_the_latch_up() {
        // The M4 in a window: heavy frames at first (14 ms), then light (4.5 ms); while the work
        // estimate still remembers the heavy ones, a light frame with 9 ms to spare is shown a
        // refresh late once.
        let display = Display {
            period: 1.0 / 60.0,
            phase: 0.0041,
            latch: 0.0172,
            gridless_from: u64::MAX,
            hiccup: Some(300),
        };
        let ran = run(&display, 1500, 1.0 / 60.0, |i| if i < 250 { 0.014 } else { 0.0045 });
        let m = median(&ran.latencies[750..]);
        assert!(m < 0.0245, "median {m}, {} misses", ran.misses);
    }
}
