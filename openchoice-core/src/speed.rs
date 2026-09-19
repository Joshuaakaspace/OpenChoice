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

use crate::catalog::Model;
use crate::fit::{Fit, RunMode};
use crate::hardware::{BandwidthSource, Hardware};

/// How a throughput number was arrived at. Travels with every estimate so it
/// can be weighed rather than trusted.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EstimateMethod {
    /// Bytes-moved over real memory bandwidth. The good case.
    Roofline,
    /// No bandwidth known for this hardware; a per-backend constant divided by
    /// parameter count. Order-of-magnitude only.
    BackendConstant,
}

impl EstimateMethod {
    pub const fn name(self) -> &'static str {
        match self {
            EstimateMethod::Roofline => "roofline",
            EstimateMethod::BackendConstant => "backend-constant",
        }
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
}

impl Speed {
    /// Whole tokens per second, for display where a decimal is noise.
    pub const fn decode_tps(&self) -> u32 {
        self.decode_tps_x10 / 10
    }
}

/// Estimate decode and prefill throughput for a model on a machine, given the
/// fit that was already computed for it.
pub fn estimate(model: &Model, hw: &Hardware, fit: &Fit, efficiency: f32) -> Speed {
    let (gpu_bw, bw_source) = hw.resolve_gpu_bandwidth();
    let ram_bw = hw.effective_ram_bandwidth();

    // Bytes that must be read to produce one token.
    let weights_bytes = fit.memory.weights_mb as f32 * 1_048_576.0;
    let kv_bytes = fit.memory.kv_cache_mb as f32 * 1_048_576.0;
    let traffic = weights_bytes + kv_bytes * 0.5;

    // Which memory that traffic comes from.
    let effective_bw = match fit.run_mode {
        RunMode::Cpu => ram_bw,
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
    }
}
