//! Static worst-case execution time (WCET) & register-spill analysis for GPU pipelines.
//!
//! Combines per-pipeline static worst-case invocation `work` (`wrela_ir::work`), local thread
//! memory (`kept` bytes), pipeline weight, and attachment bandwidth against calibrated hardware
//! tier models so unbounded loops, excessive per-thread local frames (register spills), and
//! over-budget pipelines are caught at build time before touching a GPU.

use crate::PipelineSummary;

/// Maximum per-invocation local frame / scratch memory (`kept` bytes) permitted in a shipped build.
/// Matches the GPU host safety guard (16 KiB) so stack/array spills fail at compile time instead
/// of stalling or hanging the GPU at runtime.
pub const MAX_KEPT_BYTES: u32 = 16 * 1024;

/// Local frame threshold above which a pipeline is flagged as having elevated spill risk (4 KiB).
pub const SPILL_WARN_BYTES: u32 = 4 * 1024;

/// Calibrated hardware tier cost model (femtoseconds per static IR work unit at 1080p full-screen
/// saturation, calibrated from `~/.wrela/units/Apple_M4` = 2500 fs/unit and AMD Radeon Renoir
/// ~6500 fs/unit).
#[derive(Clone, Copy, Debug)]
pub struct TierModel {
    pub name: &'static str,
    pub fs_per_work: f64,
    pub budget_ms: f64,
}

pub const TIER_M4: TierModel = TierModel { name: "apple-m4", fs_per_work: 2500.0, budget_ms: 16.6 };
pub const TIER_RADEON: TierModel =
    TierModel { name: "radeon-igpu", fs_per_work: 6500.0, budget_ms: 33.3 };
pub const TIER_BAR: TierModel =
    TierModel { name: "rtx-3060-bar", fs_per_work: 1000.0, budget_ms: 16.6 };

/// Reference 1080p pixel count (`1920 * 1080`) used to estimate full-screen worst-case pass time.
const FULL_HD_PIXELS: f64 = 1920.0 * 1080.0;

/// Static WCET summary for one GPU pipeline.
#[derive(Clone, Debug)]
pub struct PipelineWcet {
    pub entry: String,
    pub kind: &'static str,
    pub work: Option<u64>,
    pub kept: u32,
    pub weight: u64,
    pub est_1080p_ms_m4: Option<f64>,
    pub est_1080p_ms_radeon: Option<f64>,
    pub spill_risk: bool,
    pub unbounded: bool,
}

/// Whole-program static WCET and spill report across all pipelines.
#[derive(Clone, Debug)]
pub struct WcetReport {
    pub pipelines: Vec<PipelineWcet>,
    pub unbounded_count: usize,
    pub spill_risk_count: usize,
    pub over_kept_limit_count: usize,
    pub largest_kept_bytes: u32,
    pub heaviest_pipeline: Option<(String, u64)>,
    pub max_single_1080p_ms_m4: f64,
    pub max_single_1080p_ms_radeon: f64,
}

/// Estimates worst-case 1080p GPU pass time (ms) for a single invocation's static `work` on `tier`.
/// Render pipelines sum vertex + fragment work; compute pipelines use their single entry's work.
/// Note: this assumes 100% full-screen occupancy (`1920 * 1080` invocations) as a conservative
/// upper bound per pipeline (`work * fs_per_work * 1e-12 * pixels`).
pub fn estimate_1080p_ms(work: u64, tier: TierModel) -> f64 {
    (work as f64) * tier.fs_per_work * 1.0e-12 * FULL_HD_PIXELS
}

/// Analyzes every pipeline in `pipelines` for unbounded loops, spill pressure, and calibrated
/// 1080p worst-case execution time bounds.
pub fn analyze(pipelines: &[PipelineSummary]) -> WcetReport {
    let mut rows = Vec::with_capacity(pipelines.len());
    let mut unbounded_count = 0usize;
    let mut spill_risk_count = 0usize;
    let mut over_kept_limit_count = 0usize;
    let mut largest_kept_bytes = 0u32;
    let mut heaviest_pipeline: Option<(String, u64)> = None;
    let mut max_single_1080p_ms_m4 = 0.0f64;
    let mut max_single_1080p_ms_radeon = 0.0f64;

    for p in pipelines {
        let entry = p.entries.join(" + ");
        let unbounded = !p.work.is_empty() && p.work.iter().any(|w| w.is_none());
        let total_work: Option<u64> = if p.work.is_empty() {
            Some(0)
        } else if unbounded {
            None
        } else {
            Some(p.work.iter().flatten().copied().fold(0u64, u64::saturating_add))
        };
        if unbounded {
            unbounded_count += 1;
        }
        let spill_risk = p.kept >= SPILL_WARN_BYTES;
        if spill_risk {
            spill_risk_count += 1;
        }
        if p.kept > MAX_KEPT_BYTES {
            over_kept_limit_count += 1;
        }
        largest_kept_bytes = largest_kept_bytes.max(p.kept);
        let (est_m4, est_rad) = match total_work {
            Some(w) => {
                if heaviest_pipeline.as_ref().is_none_or(|(_, hw)| w > *hw) {
                    heaviest_pipeline = Some((entry.clone(), w));
                }
                let m4 = estimate_1080p_ms(w, TIER_M4);
                let rad = estimate_1080p_ms(w, TIER_RADEON);
                if p.kind == "render" {
                    max_single_1080p_ms_m4 = max_single_1080p_ms_m4.max(m4);
                    max_single_1080p_ms_radeon = max_single_1080p_ms_radeon.max(rad);
                }
                (Some(m4), Some(rad))
            }
            None => (None, None),
        };
        rows.push(PipelineWcet {
            entry,
            kind: p.kind,
            work: total_work,
            kept: p.kept,
            weight: p.weight,
            est_1080p_ms_m4: est_m4,
            est_1080p_ms_radeon: est_rad,
            spill_risk,
            unbounded,
        });
    }

    WcetReport {
        pipelines: rows,
        unbounded_count,
        spill_risk_count,
        over_kept_limit_count,
        largest_kept_bytes,
        heaviest_pipeline,
        max_single_1080p_ms_m4,
        max_single_1080p_ms_radeon,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_pipeline(
        kind: &'static str,
        name: &str,
        work: Vec<Option<u64>>,
        kept: u32,
        weight: u64,
    ) -> PipelineSummary {
        PipelineSummary {
            kind,
            entries: vec![name.to_string()],
            names: vec![name.to_string()],
            sites: vec!["main.wrela:1:1".to_string()],
            conditional: vec![false],
            attachment_bytes: None,
            work,
            unbounded: vec![String::new()],
            weight,
            kept,
            standin: None,
            stands_in: false,
            few: Vec::new(),
        }
    }

    #[test]
    fn wcet_flags_unbounded_and_spill_risks() {
        let ps = vec![
            dummy_pipeline("compute", "fast", vec![Some(500)], 256, 120),
            dummy_pipeline("render", "heavy_spill", vec![Some(10_000)], 20_000, 4000),
            dummy_pipeline("compute", "unbounded_loop", vec![None], 512, 800),
        ];
        let report = analyze(&ps);
        assert_eq!(report.unbounded_count, 1);
        assert_eq!(report.spill_risk_count, 1);
        assert_eq!(report.over_kept_limit_count, 1);
        assert_eq!(report.largest_kept_bytes, 20_000);
        assert_eq!(report.heaviest_pipeline, Some(("heavy_spill".to_string(), 10_000)));
        assert!(report.max_single_1080p_ms_m4 > 0.0);
        assert!(report.max_single_1080p_ms_radeon > report.max_single_1080p_ms_m4);
    }
}
