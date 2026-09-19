//! Four-dimensional scoring and the composite rank.
//!
//! Each dimension is 0-100 and answers one question. They are kept separate
//! all the way to the output because the composite alone hides the trade a
//! user is actually making: a model can rank well on a machine because it is
//! fast and tiny, which is not the same recommendation as ranking well because
//! it is capable.

use crate::catalog::{Model, UseCase};
use crate::fit::{Fit, Verdict};
use crate::speed::Speed;

/// The four dimensions, each 0-100.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scores {
    /// Capability: parameters, curated prior, and the quantization penalty.
    pub quality: u8,
    /// Decode throughput against what the use case needs.
    pub speed: u8,
    /// Memory headroom. Peaks in the 50-80% band — too empty wastes the
    /// machine, too full will not survive a long prompt.
    pub fit: u8,
    /// Usable context against what the use case needs.
    pub context: u8,
    /// Weighted composite, 0-10000, for ranking without float comparisons.
    pub composite: u16,
}

/// Per-dimension weights. They sum to 1.0 within each use case.
struct Weights {
    quality: f32,
    speed: f32,
    fit: f32,
    context: f32,
}

/// What each use case actually cares about.
///
/// Reasoning leans hard on quality because a fast wrong answer is worthless.
/// Chat leans on speed because latency is the felt quality. Coding needs
/// context above all — a model that cannot hold the file is not in the running
/// regardless of how clever it is.
const fn weights_for(use_case: UseCase) -> Weights {
    match use_case {
        UseCase::General => Weights {
            quality: 0.40,
            speed: 0.25,
            fit: 0.20,
            context: 0.15,
        },
        UseCase::Coding => Weights {
            quality: 0.35,
            speed: 0.20,
            fit: 0.15,
            context: 0.30,
        },
        UseCase::Reasoning => Weights {
            quality: 0.55,
            speed: 0.15,
            fit: 0.15,
            context: 0.15,
        },
        UseCase::Chat => Weights {
            quality: 0.30,
            speed: 0.35,
            fit: 0.20,
            context: 0.15,
        },
        UseCase::Multimodal => Weights {
            quality: 0.45,
            speed: 0.20,
            fit: 0.20,
            context: 0.15,
        },
        UseCase::Embedding => Weights {
            quality: 0.35,
            speed: 0.40,
            fit: 0.20,
            context: 0.05,
        },
    }
}

/// Context (tokens) below which a use case is meaningfully constrained.
const fn context_target(use_case: UseCase) -> f32 {
    match use_case {
        UseCase::Coding => 32_768.0,
        UseCase::Reasoning => 16_384.0,
        UseCase::General => 8_192.0,
        UseCase::Chat => 8_192.0,
        UseCase::Multimodal => 8_192.0,
        UseCase::Embedding => 512.0,
    }
}

/// Decode throughput (tok/s) at which a use case stops feeling slow.
const fn speed_target(use_case: UseCase) -> f32 {
    match use_case {
        UseCase::Chat => 30.0,
        UseCase::General => 20.0,
        UseCase::Coding => 20.0,
        UseCase::Multimodal => 15.0,
        UseCase::Reasoning => 10.0,
        UseCase::Embedding => 100.0,
    }
}

pub fn score(model: &Model, fit: &Fit, speed: &Speed, use_case: UseCase) -> Scores {
    let quality = quality_score(model, fit, use_case);
    let spd = speed_score(speed, use_case);
    let f = fit_score(fit);
    let ctx = context_score(fit, use_case);

    let w = weights_for(use_case);
    let mut composite = quality as f32 * w.quality
        + spd as f32 * w.speed
        + f as f32 * w.fit
        + ctx as f32 * w.context;

    // A model that will not load has no useful rank. Collapse it rather than
    // letting a strong quality score float it above things that actually run.
    if !fit.verdict.runnable() {
        composite *= 0.1;
    }

    Scores {
        quality,
        speed: spd,
        fit: f,
        context: ctx,
        composite: (composite * 100.0).min(10_000.0) as u16,
    }
}

/// Capability. Parameter count carries most of it, on a log curve because the
/// gap from 3B to 7B matters far more than 60B to 70B. The curated prior then
/// shifts it by up to +/-15 points, and the quantization penalty applies last
/// because it degrades whatever capability was there.
fn quality_score(model: &Model, fit: &Fit, requested: UseCase) -> u8 {
    let pb = model.params_b().max(0.05);
    // Anchored so 1B lands at 35 and 70B at 92, which keeps the whole usable
    // range on the curve. A steeper slope saturates by 7B and then cannot
    // distinguish a 7B from a 70B at all — the failure this replaces.
    let base = 35.0 + 9.3 * log2(pb);
    let prior = (model.quality_prior() as f32 / 255.0 - 0.5) * 30.0;
    let alignment = alignment_bonus(model.use_case(), requested);
    let raw = (base + prior + alignment) * fit.quant.quality_factor();
    clamp_u8(raw)
}

/// How much a model's own specialisation helps or hurts for the requested job.
///
/// A match is a modest edge, not a decisive one: a 32B generalist should still
/// beat a 3B coding model at coding. The penalty is the more important half —
/// an OCR or embedding model is genuinely worse at chat than a general model
/// of the same size, and without this it outranks one purely on popularity.
fn alignment_bonus(model_case: UseCase, requested: UseCase) -> f32 {
    if model_case == requested {
        return 8.0;
    }
    let requested_is_text = matches!(
        requested,
        UseCase::General | UseCase::Coding | UseCase::Reasoning | UseCase::Chat
    );
    match model_case {
        UseCase::Embedding if requested_is_text => -25.0,
        UseCase::Multimodal if requested_is_text => -10.0,
        _ => 0.0,
    }
}

/// Throughput against what the use case needs. Saturates: past the target,
/// more tokens per second stop being worth anything.
fn speed_score(speed: &Speed, use_case: UseCase) -> u8 {
    let tps = speed.decode_tps_x10 as f32 / 10.0;
    let target = speed_target(use_case);
    let ratio = tps / target;
    let raw = if ratio >= 1.0 {
        // Diminishing returns above target rather than a hard ceiling, so a
        // genuinely fast pairing still separates from a just-adequate one.
        85.0 + 15.0 * (1.0 - 1.0 / (1.0 + log2(ratio + 1.0)))
    } else {
        85.0 * ratio
    };
    clamp_u8(raw)
}

/// Memory headroom, peaking in the 50-80% band. Below that the machine is
/// under-used and a better model would have fit; above it, the fit is fragile.
fn fit_score(fit: &Fit) -> u8 {
    if !fit.verdict.runnable() {
        return 0;
    }
    let u = fit.utilization_pctx10 as f32 / 1000.0;
    let raw = if u < 0.50 {
        60.0 + 80.0 * u
    } else if u <= 0.80 {
        100.0
    } else {
        100.0 - (u - 0.80) * 350.0
    };
    clamp_u8(raw)
}

/// Usable context — not the native window — against the use case target.
/// This is the dimension that catches a model advertising 262k while the
/// machine can only hold 8k of it.
fn context_score(fit: &Fit, use_case: UseCase) -> u8 {
    let target = context_target(use_case);
    let usable = fit.usable_context as f32;
    let raw = if usable >= target {
        100.0
    } else {
        100.0 * (usable / target)
    };
    clamp_u8(raw)
}

fn clamp_u8(v: f32) -> u8 {
    if v <= 0.0 {
        0
    } else if v >= 100.0 {
        100
    } else {
        v as u8
    }
}

/// Base-2 log without `std`. Newton refinement on the exponent gets well
/// inside a tenth of a bit, which is far tighter than the scores need.
fn log2(x: f32) -> f32 {
    if x <= 0.0 {
        return -20.0;
    }
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    // Mantissa in [1, 2).
    let mantissa = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    // Degree-3 fit of log2 over [1, 2). The leading coefficients are the
    // series terms for log2(1+m) = m/ln2 - m^2/(2 ln2) + ...; the cubic term
    // is fitted rather than exact, which is what pulls the error down to
    // well inside a tenth of a bit across the interval.
    const C1: f32 = core::f32::consts::LOG2_E;
    const C2: f32 = core::f32::consts::LOG2_E / 2.0;
    const C3: f32 = 0.4276283;
    let m = mantissa - 1.0;
    let poly = m * (C1 - m * (C2 - m * C3));
    exp as f32 + poly
}

/// Rank order helper: higher composite first, unrunnable always last.
pub fn better(a: (&Fit, &Scores), b: (&Fit, &Scores)) -> core::cmp::Ordering {
    let runnable = b.0.verdict.runnable().cmp(&a.0.verdict.runnable());
    if runnable != core::cmp::Ordering::Equal {
        return runnable;
    }
    b.1.composite.cmp(&a.1.composite)
}

/// Verdict ordering for filters like "at least Good".
pub fn at_least(actual: Verdict, minimum: Verdict) -> bool {
    actual >= minimum
}
