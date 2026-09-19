//! Memory model, run-mode selection, and the fit verdict.
//!
//! The verdict is a pure function of one number — how full the run mode's
//! memory pool is — and is then capped by what the execution path can actually
//! deliver. Nothing else feeds into it. That constraint is deliberate: a
//! verdict assembled from several partly-overlapping signals is one nobody can
//! reproduce or argue with.

use crate::catalog::{flags, Model, UseCase};
use crate::hardware::Hardware;
use crate::quant::{KvQuant, Quant};

/// Bytes in a MiB, as a float, for the memory arithmetic.
const MIB: f32 = 1_048_576.0;

/// How comfortably a model sits in the memory it would run from.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum Verdict {
    /// Will not load, or will thrash badly enough that it may as well not.
    TooTight = 0,
    /// Loads, but with nothing spare. Long prompts will push it over.
    Marginal = 1,
    /// Comfortable.
    Good = 2,
    /// Comfortable and on the accelerator.
    Perfect = 3,
}

impl Verdict {
    pub const fn name(self) -> &'static str {
        match self {
            Verdict::TooTight => "Too Tight",
            Verdict::Marginal => "Marginal",
            Verdict::Good => "Good",
            Verdict::Perfect => "Perfect",
        }
    }

    pub const fn runnable(self) -> bool {
        !matches!(self, Verdict::TooTight)
    }
}

/// Which memory the weights actually live in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum RunMode {
    /// Entirely in accelerator memory.
    Gpu = 0,
    /// Sparse model with the inactive experts parked in system RAM.
    MoeOffload = 1,
    /// Split across accelerator and system memory.
    CpuGpu = 2,
    /// Entirely in system RAM.
    Cpu = 3,
}

impl RunMode {
    pub const fn name(self) -> &'static str {
        match self {
            RunMode::Gpu => "GPU",
            RunMode::MoeOffload => "MoE offload",
            RunMode::CpuGpu => "CPU+GPU",
            RunMode::Cpu => "CPU",
        }
    }

    /// The ceiling this path can reach.
    ///
    /// Only a model sitting wholly on the accelerator can be Perfect. The
    /// others cap at Good rather than being pushed down to Marginal: a model
    /// that fits comfortably in RAM is genuinely runnable, just not fast.
    pub const fn verdict_cap(self) -> Verdict {
        match self {
            RunMode::Gpu => Verdict::Perfect,
            _ => Verdict::Good,
        }
    }
}

/// Knobs the caller can turn. Defaults are the ones the estimator was
/// calibrated against; changing them changes the answers.
#[derive(Clone, Copy, Debug)]
pub struct Opts {
    /// Cap the context used for the memory estimate. `None` uses the model's
    /// native window, which is what makes long-context models look expensive
    /// — correctly so.
    pub context: Option<u32>,
    pub kv_quant: KvQuant,
    /// Fraction of theoretical bandwidth a real decode loop achieves.
    pub efficiency: f32,
    /// Allow parking inactive MoE experts in system RAM.
    pub allow_moe_offload: bool,
    /// Highest quantization to consider.
    ///
    /// Defaults to `Q8_0`, not `F16`. Full precision costs roughly double the
    /// memory of Q8_0 to buy a 0.2% quality difference, so letting it win the
    /// hierarchy walk systematically recommends a small model at F16 over a
    /// much more capable one at Q8 — the wrong trade on every machine anyone
    /// actually owns. Pass `F16` explicitly to consider it.
    pub max_quant: Quant,
    /// Score against this use case's priorities rather than the one the
    /// catalog assigned the model. This is what `--use-case coding` should
    /// mean: reweight the ranking, not hide every model not already tagged
    /// as a coding model.
    pub use_case: Option<UseCase>,
}

impl Default for Opts {
    fn default() -> Opts {
        Opts {
            context: None,
            kv_quant: KvQuant::F16,
            efficiency: 0.55,
            allow_moe_offload: true,
            max_quant: Quant::Q8_0,
            use_case: None,
        }
    }
}

/// Where the KV-cache figure came from. A caller that reports a memory total
/// without reporting this is passing off an inference as a measurement.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvSource {
    /// Layer and head counts came from the model config.
    Metadata,
    /// Counts were inferred from the parameter count via the size-class table.
    Estimated,
}

/// The memory breakdown behind a verdict, in MiB.
#[derive(Clone, Copy, Debug)]
pub struct MemoryBreakdown {
    pub weights_mb: u32,
    pub kv_cache_mb: u32,
    pub overhead_mb: u32,
    /// Inactive MoE experts pushed out to system RAM, if any.
    pub offloaded_mb: u32,
    pub kv_source: KvSource,
}

impl MemoryBreakdown {
    pub const fn resident_mb(&self) -> u32 {
        self.weights_mb + self.kv_cache_mb + self.overhead_mb
    }
}

/// The result of scoring one model against one machine.
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub verdict: Verdict,
    pub run_mode: RunMode,
    pub quant: Quant,
    pub memory: MemoryBreakdown,
    /// Memory the run mode draws from, MiB.
    pub pool_mb: u32,
    /// `resident / pool`, in tenths of a percent, so it survives the trip
    /// through a `no_std` formatter without float printing.
    pub utilization_pctx10: u16,
    /// Context that actually fits after weights and overhead, capped at the
    /// model's native window. A Perfect fit with an 8k usable window out of a
    /// 262k one is a very different proposition for real work.
    pub usable_context: u32,
    /// Context the memory figures were computed at.
    pub effective_context: u32,
}

/// Layer count, KV head count, and head dimension by parameter size class.
///
/// Used only when the catalog lacks real architecture metadata. The values
/// track the dominant GQA designs at each size — Llama 3 8B is (32, 8, 128),
/// 70B is (80, 8, 128), Qwen 2.5 32B is (64, 8, 128) — so the estimate is a
/// reasonable central case rather than a guess. Anything built on it is
/// flagged `KvSource::Estimated`.
const SIZE_CLASSES: &[(f32, u16, u16, u16)] = &[
    // max params_b, layers, kv_heads, head_dim
    (0.6, 24, 4, 64),
    (2.0, 28, 4, 128),
    (4.0, 32, 8, 128),
    (9.0, 32, 8, 128),
    (16.0, 40, 8, 128),
    (24.0, 48, 8, 128),
    (40.0, 64, 8, 128),
    (80.0, 80, 8, 128),
];

fn arch_for(model: &Model) -> (u16, u16, u16, KvSource) {
    if model.has_flag(flags::ARCH_METADATA)
        && model.num_layers() > 0
        && model.num_kv_heads() > 0
        && model.head_dim() > 0
    {
        return (
            model.num_layers(),
            model.num_kv_heads(),
            model.head_dim(),
            KvSource::Metadata,
        );
    }
    let pb = model.params_b();
    for &(max_b, layers, kv, hd) in SIZE_CLASSES {
        if pb <= max_b {
            return (layers, kv, hd, KvSource::Estimated);
        }
    }
    (96, 8, 128, KvSource::Estimated)
}

/// Weight footprint in MiB at a given quantization.
pub fn weights_mb(params_m: u32, quant: Quant) -> u32 {
    let bytes = params_m as f32 * 1.0e6 * quant.bits_per_weight() / 8.0;
    (bytes / MIB) as u32
}

/// KV-cache bytes per token: two tensors (K and V) per layer, each
/// `kv_heads * head_dim` elements wide.
fn kv_bytes_per_token(layers: u16, kv_heads: u16, head_dim: u16, kv: KvQuant) -> f32 {
    2.0 * layers as f32 * kv_heads as f32 * head_dim as f32 * kv.bytes_per_element()
}

/// Runtime overhead in MiB: compute buffers, the graph, and allocator slack.
///
/// Scales with the model because the activation buffers do, and with context
/// because the attention scratch does.
fn overhead_mb(weights: u32, context: u32) -> u32 {
    128 + (weights as f32 * 0.05) as u32 + (context / 512)
}

/// Score one model against one machine.
///
/// Walks the quantization hierarchy best-quality-first and returns the first
/// format that actually fits; if none does at the requested context, retries
/// at half context before giving up. The returned `Fit` always describes a
/// real configuration — when nothing fits, it describes the least-bad one and
/// says `TooTight`.
pub fn evaluate(model: &Model, hw: &Hardware, opts: &Opts) -> Fit {
    let native_ctx = if model.context_length() == 0 {
        4096
    } else {
        model.context_length()
    };
    let requested = opts.context.unwrap_or(native_ctx).min(native_ctx).max(512);

    for &ctx in &[requested, requested / 2] {
        if ctx < 512 {
            continue;
        }
        for quant in Quant::HIERARCHY {
            if quant < opts.max_quant || !model.supports_quant(quant) {
                continue;
            }
            let fit = evaluate_at(model, hw, opts, quant, ctx);
            if fit.verdict.runnable() {
                return fit;
            }
        }
    }

    // Nothing fits. Report the most compressed configuration so the caller can
    // see how far off it is rather than just being told "no".
    evaluate_at(model, hw, opts, Quant::Q2K, requested)
}

/// Evaluate one specific (quantization, context) pairing.
pub fn evaluate_at(model: &Model, hw: &Hardware, opts: &Opts, quant: Quant, context: u32) -> Fit {
    let (layers, kv_heads, head_dim, kv_source) = arch_for(model);
    let per_token = kv_bytes_per_token(layers, kv_heads, head_dim, opts.kv_quant);

    let w_full = weights_mb(model.params_m(), quant);
    let w_active = weights_mb(model.active_params_m(), quant);
    let kv = ((per_token * context as f32) / MIB) as u32;
    let oh = overhead_mb(w_full, context);

    let usable_ram = hw.usable_ram_mb();
    let vram = hw.vram_mb;
    let full_resident = w_full + kv + oh;

    // Pick the run mode: the fastest path whose pool can hold the model.
    let (run_mode, pool, resident, offloaded) = if hw.backend.is_cpu() || vram == 0 {
        (RunMode::Cpu, usable_ram, full_resident, 0)
    } else if full_resident <= vram {
        (RunMode::Gpu, vram, full_resident, 0)
    } else if model.is_moe() && opts.allow_moe_offload && !hw.unified {
        let resident = w_active + kv + oh;
        let offload = w_full.saturating_sub(w_active);
        if resident <= vram && offload <= usable_ram {
            (RunMode::MoeOffload, vram, resident, offload)
        } else {
            (RunMode::CpuGpu, vram + usable_ram, full_resident, 0)
        }
    } else if hw.unified {
        // VRAM and RAM are the same silicon. Spilling buys nothing, so the
        // only honest fallback is the CPU path against the same pool.
        (RunMode::Cpu, usable_ram, full_resident, 0)
    } else {
        (RunMode::CpuGpu, vram + usable_ram, full_resident, 0)
    };

    let ratio = if pool == 0 {
        f32::INFINITY
    } else {
        resident as f32 / pool as f32
    };
    let verdict = cap(pure_verdict(ratio), run_mode);

    // How much context the pool would actually take at this quantization.
    let spare = pool.saturating_sub(w_full + oh) as f32 * MIB;
    let usable_context = if per_token > 0.0 {
        ((spare / per_token) as u32).min(model.context_length().max(512))
    } else {
        context
    };

    Fit {
        verdict,
        run_mode,
        quant,
        memory: MemoryBreakdown {
            weights_mb: if run_mode == RunMode::MoeOffload {
                w_active
            } else {
                w_full
            },
            kv_cache_mb: kv,
            overhead_mb: oh,
            offloaded_mb: offloaded,
            kv_source,
        },
        pool_mb: pool,
        utilization_pctx10: (ratio * 1000.0).min(u16::MAX as f32) as u16,
        usable_context,
        effective_context: context,
    }
}

/// The verdict band. Pool utilization in, verdict out, nothing else.
///
/// The top band stops at 98% rather than 100% because a pool filled to the
/// last percent leaves nothing for allocator slack or fragmentation, and does
/// not load in practice.
fn pure_verdict(ratio: f32) -> Verdict {
    if ratio <= 0.60 {
        Verdict::Perfect
    } else if ratio <= 0.85 {
        Verdict::Good
    } else if ratio <= 0.98 {
        Verdict::Marginal
    } else {
        Verdict::TooTight
    }
}

fn cap(level: Verdict, mode: RunMode) -> Verdict {
    let ceiling = mode.verdict_cap();
    if level > ceiling {
        ceiling
    } else {
        level
    }
}
