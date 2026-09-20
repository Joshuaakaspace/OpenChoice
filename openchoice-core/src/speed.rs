//! Throughput estimation.
//!
//! Decode and prefill are bound by different resources and are estimated
//! separately. Conflating them is the single most common way a "tokens per
//! second" number ends up meaningless.
//!
//! **Decode** is memory-bandwidth-bound: generating one token requires reading
//! every resident weight once. Throughput is therefore bytes-moved over
//! bandwidth, derated for the fraction of theoretical bandwidth a real kernel
//! achieves.
//!
//! **Prefill** is compute-bound: roughly `2 * active_params` FLOPs per prompt
//! token. Bandwidth says nothing useful about it, so it is estimated from fp16
//! matmul throughput or not at all. When that is unknown, prefill and TTFT are
//! `None` — deliberately different from `0.0`, which would read as
//! "immeasurably slow" rather than "not estimated".

use crate::catalog::{Catalog, Measurement, Model};
use crate::fit::{Fit, RunMode};
use crate::hardware::{BandwidthSource, Hardware};

/// How a throughput number was arrived at. Travels with every estimate so it
/// can be weighed rather than trusted.
///
/// Ordered best to worst. The distinction that matters most is the first one:
/// a measurement and a formula are both reported in tokens per second, and
/// presenting them identically is how a guess acquires unearned authority.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum EstimateMethod {
    /// Somebody ran this model, at this quantization, on this hardware and
    /// recorded the result. Not an estimate at all.
    Measured,
    /// Measured on this hardware, but at a different quantization than the one
    /// being reported, and rescaled by the change in weight bytes. Decode is
    /// bandwidth-bound, so that rescaling is sound — but it is no longer a
    /// number anybody observed, and it does not claim to be.
    MeasuredAdjusted,
    /// A formula, scaled by a factor derived from real measurements taken on
    /// this same hardware with other models.
    Calibrated,
    /// Bytes-moved over real memory bandwidth, with nothing measured behind it.
    Roofline,
    /// No bandwidth known for this hardware either; a per-backend constant
    /// divided by parameter count. Order-of-magnitude only.
    BackendConstant,
}

impl EstimateMethod {
    pub const fn name(self) -> &'static str {
        match self {
            EstimateMethod::Measured => "measured",
            EstimateMethod::MeasuredAdjusted => "measured-adjusted",
            EstimateMethod::Calibrated => "calibrated",
            EstimateMethod::Roofline => "roofline",
            EstimateMethod::BackendConstant => "backend-constant",
        }
    }

    /// Whether this number came from hardware rather than arithmetic.
    pub const fn is_measured(self) -> bool {
        matches!(self, EstimateMethod::Measured)
    }
}

/// Everything needed to reproduce a throughput estimate by hand.
#[derive(Clone, Copy, Debug)]
pub struct Speed {
    /// Decode throughput, tokens/second x10 (integer, so it formats without
    /// float support on targets that lack it).
    pub decode_tps_x10: u32,
    /// Prompt-processing throughput, tokens/second x10. `None` when fp16
    /// throughput for this hardware is unknown.
    pub prefill_tps_x10: Option<u32>,
    /// Time to first token in milliseconds for a prompt filling the effective
    /// context. `None` under the same condition as `prefill_tps_x10`.
    pub ttft_ms: Option<u32>,
    pub method: EstimateMethod,
    /// The bandwidth figure actually used, GB/s.
    pub bandwidth_gbps: u16,
    pub bandwidth_source: BandwidthSource,
    pub efficiency: f32,
    /// The measurement behind this number, when there is one.
    pub measurement: Option<Measurement>,
    /// The correction factor applied, and how many measurements it came from.
    /// Only set for `Calibrated`.
    pub calibration: Option<(f32, u8)>,
}

impl Speed {
    /// Whole tokens per second, for display where a decimal is noise.
    pub const fn decode_tps(&self) -> u32 {
        self.decode_tps_x10 / 10
    }
}

/// Decode and prefill throughput for a model on a machine.
///
/// Consults the catalog's measurements first. The order is deliberate and the
/// steps are never blended:
///
/// 1. **Measured** — somebody ran exactly this on exactly this hardware.
/// 2. **Calibrated** — the formula, scaled by how wrong it has been on this
///    hardware for other models.
/// 3. **Roofline** — the formula, unadjusted.
/// 4. **Backend constant** — a last resort when even bandwidth is unknown.
pub fn estimate_with_catalog(
    catalog: &Catalog<'_>,
    model: &Model,
    hw: &Hardware,
    fit: &Fit,
    efficiency: f32,
) -> Speed {
    let mut speed = estimate(model, hw, fit, efficiency);

    if let Some(m) = catalog.measurement(hw.hw_key, model.index()) {
        speed.measurement = Some(m);
        match m.quant {
            // Same quantization: a real number, reported as-is. It is not
            // averaged with the formula, because the average of a measurement
            // and a guess is neither of them.
            Some(q) if q == fit.quant => {
                speed.decode_tps_x10 = m.tps_x10 as u32;
                speed.method = EstimateMethod::Measured;
            }
            // Different quantization. The run is still informative — it is the
            // same weights on the same silicon — but it moved a different
            // number of bytes per token. Rescale by that ratio rather than
            // reporting a Q4 result against a Q8 row, which would have shown
            // a 39 tok/s measurement next to a 2 tok/s configuration.
            Some(q) => {
                let ratio = q.bits_per_weight() / fit.quant.bits_per_weight();
                speed.decode_tps_x10 = (m.tps_x10 as f32 * ratio) as u32;
                speed.method = EstimateMethod::MeasuredAdjusted;
            }
            None => {}
        }
        // The measured TTFT belongs to whatever prompt the benchmark used, and
        // that length was not recorded. Attaching it to this row's context
        // would be asserting something nobody measured, so `ttft_ms` keeps the
        // formula's value and the raw figure stays on `measurement` for a
        // caller that wants to show it as what it is.
        if speed.method != EstimateMethod::Roofline {
            return speed;
        }
    }

    // No measurement for this pairing, but if the formula has been checked
    // against this hardware before, apply what that showed. A card the
    // roofline overshoots by 30% on every model measured is overshooting the
    // rest of the catalog by roughly 30% too.
    if speed.method == EstimateMethod::Roofline {
        if let Some((factor, samples)) = catalog.calibration(hw.hw_key) {
            speed.decode_tps_x10 = (speed.decode_tps_x10 as f32 * factor) as u32;
            speed.method = EstimateMethod::Calibrated;
            speed.calibration = Some((factor, samples));
            if let Some(ttft) = speed.ttft_ms {
                speed.ttft_ms = Some((ttft as f32 / factor.max(0.05)) as u32);
            }
        }
    }

    speed
}

/// The formula alone, with no measurement lookup. Exposed for callers that
/// deliberately want the unadjusted estimate — comparing it against a
/// measurement, for instance.
pub fn estimate(model: &Model, hw: &Hardware, fit: &Fit, efficiency: f32) -> Speed {
    let (gpu_bw, bw_source) = hw.resolve_gpu_bandwidth();
    let ram_bw = hw.effective_ram_bandwidth();

    // Bytes that must be read to produce one token.
    let weights_bytes = fit.memory.weights_mb as f32 * 1_048_576.0;
    let kv_bytes = fit.memory.kv_cache_mb as f32 * 1_048_576.0;
    let traffic = weights_bytes + kv_bytes * 0.5;

    // Which memory that traffic comes from.
    let effective_bw = match fit.run_mode {
        // Both read system memory; on a unified machine that is the same fast
        // pool the GPU uses, which `effective_ram_bandwidth` already accounts
        // for, so the two cases share an arm rather than a magic constant.
        RunMode::Cpu | RunMode::UnifiedSpill => ram_bw,
        RunMode::Gpu => {
            if gpu_bw > 0.0 {
                gpu_bw
            } else {
                0.0
            }
        }
        RunMode::MoeOffload => {
            // Active experts are resident on the accelerator; the router still
            // touches RAM for the misses. Weight the pools by where the bytes
            // live for a typical token.
            if gpu_bw > 0.0 {
                gpu_bw * 0.85 + ram_bw * 0.15
            } else {
                0.0
            }
        }
        RunMode::CpuGpu => {
            // Harmonic-style blend: the slow pool dominates, because the whole
            // layer stack must be walked in order and the CPU half gates it.
            let total = fit.memory.resident_mb() as f32;
            if total <= 0.0 || gpu_bw <= 0.0 {
                ram_bw
            } else {
                let gpu_frac = (hw.vram_mb as f32 / total).min(1.0);
                let cpu_frac = 1.0 - gpu_frac;
                1.0 / (gpu_frac / gpu_bw + cpu_frac / ram_bw)
            }
        }
    };

    let (decode_x10, method) = if effective_bw > 0.0 && traffic > 0.0 {
        let tps = (effective_bw * 1.0e9 / traffic) * efficiency;
        ((tps * 10.0) as u32, EstimateMethod::Roofline)
    } else {
        // Nothing known about this hardware's memory system. Fall back to a
        // per-backend constant over parameter count and say so.
        let pb = model.active_params_b().max(0.05);
        let tps = hw.backend.fallback_k() / pb * fit.quant.speed_multiplier();
        ((tps * 10.0) as u32, EstimateMethod::BackendConstant)
    };

    // Prefill: compute-bound, so only estimable with fp16 throughput.
    let (prefill_x10, ttft) = if hw.tflops_fp16_x10 > 0 {
        let tflops = hw.tflops_fp16_x10 as f32 / 10.0;
        let flops_per_token = 2.0 * model.active_params_m() as f32 * 1.0e6;
        if flops_per_token > 0.0 {
            // Prefill sustains a higher fraction of peak than decode does:
            // it is a dense GEMM, not a bandwidth-starved GEMV.
            let tps = (tflops * 1.0e12 / flops_per_token) * 0.35;
            let ms = if tps > 0.0 {
                (fit.effective_context as f32 / tps * 1000.0) as u32
            } else {
                0
            };
            (Some((tps * 10.0) as u32), Some(ms))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    Speed {
        decode_tps_x10: decode_x10,
        prefill_tps_x10: prefill_x10,
        ttft_ms: ttft,
        method,
        bandwidth_gbps: effective_bw as u16,
        bandwidth_source: if effective_bw > 0.0 {
            bw_source
        } else {
            BandwidthSource::BackendConstant
        },
        efficiency,
        measurement: None,
        calibration: None,
    }
}
