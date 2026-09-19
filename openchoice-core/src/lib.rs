//! # openchoice-core
//!
//! Answers one question: **will this model run on this hardware, and how
//! well?** It is `no_std`, allocation-free, and has no dependencies, so the
//! same arithmetic runs in a desktop CLI and on a microcontroller with a few
//! hundred kilobytes of RAM.
//!
//! The design constraint that shapes everything here is that the catalog is
//! read in place from flash rather than parsed into memory. A 15,000-model
//! catalog is ~1 MB of packed records and strings; nothing is ever copied out
//! of it, and a ranking pass over the whole thing allocates nothing.
//!
//! ```no_run
//! use openchoice_core::{Catalog, Hardware, Backend, Opts, evaluate_model};
//!
//! let bytes: &[u8] = &[]; // include_bytes!("../../catalog/openchoice.ocb")
//! let catalog = Catalog::parse(bytes).unwrap();
//! let hw = Hardware {
//!     ram_mb: 32 * 1024,
//!     vram_mb: 24 * 1024,
//!     unified: false,
//!     backend: Backend::Cuda,
//!     gpu_bandwidth_gbps: 1008,
//!     ram_bandwidth_gbps: 0,
//!     tflops_fp16_x10: 1654,
//!     os_reserve_mb: 2048,
//! };
//!
//! if let Some(model) = catalog.find("mistral-7b") {
//!     let r = evaluate_model(&model, &hw, &Opts::default());
//!     // r.fit.verdict, r.speed.decode_tps(), r.scores.composite
//! }
//! ```

#![no_std]
#![forbid(unsafe_code)]

pub mod catalog;
pub mod fit;
pub mod hardware;
pub mod quant;
pub mod score;
pub mod speed;

pub use catalog::{Catalog, CatalogError, Model, UseCase};
pub use fit::{Fit, KvSource, MemoryBreakdown, Opts, RunMode, Verdict};
pub use hardware::{Backend, BandwidthSource, Hardware};
pub use quant::{KvQuant, Quant};
pub use score::Scores;
pub use speed::{EstimateMethod, Speed};

#[cfg(feature = "gpu-db")]
pub use hardware::lookup_gpu;

/// Everything known about one model on one machine.
#[derive(Clone, Copy, Debug)]
pub struct Recommendation<'a> {
    pub model: Model<'a>,
    pub fit: Fit,
    pub speed: Speed,
    pub scores: Scores,
}

impl core::fmt::Debug for Model<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Model")
            .field("name", &self.name())
            .field("params_m", &self.params_m())
            .field("context_length", &self.context_length())
            .finish()
    }
}

/// Score one model. The three stages run in order because each depends on the
/// last: the quantization and run mode chosen by the fit determine how many
/// bytes move per token, which determines the speed, which feeds the score.
pub fn evaluate_model<'a>(model: &Model<'a>, hw: &Hardware, opts: &Opts) -> Recommendation<'a> {
    let fit = fit::evaluate(model, hw, opts);
    let speed = speed::estimate(model, hw, &fit, opts.efficiency);
    // The caller's use case wins when given: asking "what should I use for
    // coding" must reweight the whole catalog, not just re-sort the models
    // already labelled as coding models.
    let use_case = opts.use_case.unwrap_or_else(|| model.use_case());
    let scores = score::score(model, &fit, &speed, use_case);
    Recommendation {
        model: *model,
        fit,
        speed,
        scores,
    }
}

/// A fixed-capacity best-N collector.
///
/// Ranking the whole catalog needs a sort, and a sort needs somewhere to put
/// the results. On a microcontroller that cannot be a heap-allocated `Vec`, so
/// this keeps the best `N` seen so far in a caller-owned array via insertion.
/// One pass over 15,000 models with `N = 20` costs nothing measurable and
/// allocates nothing at all.
pub struct TopN<'a, const N: usize> {
    items: [Option<Recommendation<'a>>; N],
    len: usize,
}

impl<'a, const N: usize> Default for TopN<'a, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, const N: usize> TopN<'a, N> {
    pub const fn new() -> Self {
        TopN {
            items: [None; N],
            len: 0,
        }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert if it ranks; drop it if it does not. Keeps the array sorted best
    /// first, so the common case (a model that does not make the cut) exits
    /// after a single comparison against the tail.
    ///
    /// Repackages of weights already held are collapsed rather than added: a
    /// top-10 that is eight spellings of one model is not a top-10.
    pub fn push(&mut self, rec: Recommendation<'a>) {
        if let Some(slot) = self.find_same_weights(&rec) {
            let incumbent = self.items[slot].as_ref().expect("occupied slot");
            if ranks_above(&rec, incumbent) {
                self.items[slot] = Some(rec);
                self.resort_from(slot);
            }
            return;
        }

        if self.len == N {
            if let Some(worst) = self.items[N - 1].as_ref() {
                if !ranks_above(&rec, worst) {
                    return;
                }
            }
        }

        let mut at = self.len.min(N - 1);
        while at > 0 {
            match self.items[at - 1].as_ref() {
                Some(prev) if ranks_above(&rec, prev) => {
                    self.items[at] = self.items[at - 1];
                    at -= 1;
                }
                _ => break,
            }
        }
        self.items[at] = Some(rec);
        if self.len < N {
            self.len += 1;
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Recommendation<'a>> {
        self.items[..self.len].iter().filter_map(|s| s.as_ref())
    }

    pub fn get(&self, i: usize) -> Option<&Recommendation<'a>> {
        self.items.get(i)?.as_ref()
    }

    fn find_same_weights(&self, rec: &Recommendation<'a>) -> Option<usize> {
        let name = rec.model.name();
        (0..self.len).find(|&i| {
            self.items[i]
                .as_ref()
                .is_some_and(|held| catalog::same_weights(held.model.name(), name))
        })
    }

    /// Bubble the entry at `slot` back toward the front after it improved.
    fn resort_from(&mut self, slot: usize) {
        let mut at = slot;
        while at > 0 {
            let (Some(cur), Some(prev)) = (self.items[at], self.items[at - 1]) else {
                break;
            };
            if !ranks_above(&cur, &prev) {
                break;
            }
            self.items.swap(at, at - 1);
            at -= 1;
        }
    }
}

fn ranks_above(a: &Recommendation, b: &Recommendation) -> bool {
    score::better((&a.fit, &a.scores), (&b.fit, &b.scores)) == core::cmp::Ordering::Less
}

/// Filters applied while ranking. Every field is optional; the default keeps
/// everything runnable.
#[derive(Clone, Copy, Debug, Default)]
pub struct Filter {
    pub use_case: Option<UseCase>,
    pub min_verdict: Option<Verdict>,
    /// Skip anything below this decode throughput, tok/s.
    pub min_tps: Option<u32>,
    /// Skip anything whose usable context falls short of this.
    pub min_context: Option<u32>,
    /// Skip models larger than this, in millions of parameters.
    pub max_params_m: Option<u32>,
}

impl Filter {
    fn admits(&self, rec: &Recommendation) -> bool {
        if let Some(uc) = self.use_case {
            if rec.model.use_case() != uc {
                return false;
            }
        }
        if let Some(v) = self.min_verdict {
            if rec.fit.verdict < v {
                return false;
            }
        } else if !rec.fit.verdict.runnable() {
            return false;
        }
        if let Some(t) = self.min_tps {
            if rec.speed.decode_tps() < t {
                return false;
            }
        }
        if let Some(c) = self.min_context {
            if rec.fit.usable_context < c {
                return false;
            }
        }
        if let Some(p) = self.max_params_m {
            if rec.model.params_m() > p {
                return false;
            }
        }
        true
    }
}

/// Rank an entire catalog against one machine, keeping the best `N`.
///
/// This is the whole product in one function: one pass, no allocation, and the
/// caller owns the output buffer.
pub fn recommend<'a, const N: usize>(
    catalog: &Catalog<'a>,
    hw: &Hardware,
    opts: &Opts,
    filter: &Filter,
) -> TopN<'a, N> {
    let mut top = TopN::<N>::new();
    for model in catalog.iter() {
        let rec = evaluate_model(&model, hw, opts);
        if filter.admits(&rec) {
            top.push(rec);
        }
    }
    top
}
