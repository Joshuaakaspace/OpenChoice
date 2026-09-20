//! Zero-copy reader for the packed `.ocb` catalog.
//!
//! The upstream project this engine descends from ships its catalog as a
//! 13.4 MB JSON blob embedded with `include_str!` and parsed at startup. That
//! is fine on a laptop and impossible on a microcontroller. The same 15,000
//! models pack into fixed-width 32-byte records plus a deduplicated string
//! table, which is small enough to sit in flash and be read in place: no
//! parser, no heap, no copy. A `Catalog` borrows a `&[u8]` and every accessor
//! decodes straight out of it.
//!
//! Format is little-endian throughout. See `docs/catalog-format.md`.

use crate::quant::Quant;

/// `"OCB1"` — OpenChoice Binary catalog.
pub const MAGIC: [u8; 4] = *b"OCB1";
/// Version 2 adds the measurement and calibration sections.
pub const FORMAT_VERSION: u16 = 2;
pub const HEADER_LEN: usize = 48;
pub const RECORD_LEN: usize = 32;
/// Stride of one measurement record.
pub const MEASUREMENT_LEN: usize = 16;
/// Stride of one calibration record.
pub const CALIBRATION_LEN: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CatalogError {
    TooShort,
    BadMagic,
    UnsupportedVersion(u16),
    /// The record stride in the header is not what this build knows how to
    /// decode. Rejected rather than guessed at.
    BadRecordSize(u16),
    /// A section offset or length runs past the end of the buffer.
    Truncated,
}

/// Stable identifier for a machine, used to attach real measurements to it.
///
/// This is the one piece of string handling that has to agree between the
/// packer (which reads hardware names out of submitted benchmark files) and
/// whatever is asking the question later. Keeping it to a single tiny function
/// — lowercase, drop everything that is not alphanumeric, FNV-1a — means there
/// is one definition to keep honest rather than a normalizer on each side that
/// can quietly drift apart.
///
/// `"NVIDIA GeForce RTX 4090"` and `"nvidia geforce rtx-4090"` hash alike;
/// `"RTX 4080"` does not.
pub fn hw_key(name: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for &b in name.as_bytes() {
        if !b.is_ascii_alphanumeric() {
            continue;
        }
        hash ^= b.to_ascii_lowercase() as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    // Zero is reserved for "no hardware identity known", so a name that
    // genuinely hashes there is nudged off it.
    if hash == 0 {
        1
    } else {
        hash
    }
}

/// Which runtime produced a measurement. Throughput differs enough between
/// them that the number is not interpretable without it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Provider {
    Unknown = 0,
    LlamaCpp = 1,
    Ollama = 2,
    Mlx = 3,
    Vllm = 4,
}

impl Provider {
    pub const fn name(self) -> &'static str {
        match self {
            Provider::Unknown => "unknown",
            Provider::LlamaCpp => "llama.cpp",
            Provider::Ollama => "ollama",
            Provider::Mlx => "mlx",
            Provider::Vllm => "vllm",
        }
    }

    pub const fn from_u8(v: u8) -> Provider {
        match v {
            1 => Provider::LlamaCpp,
            2 => Provider::Ollama,
            3 => Provider::Mlx,
            4 => Provider::Vllm,
            _ => Provider::Unknown,
        }
    }
}

/// A real throughput measurement someone recorded on real hardware.
///
/// These are the ground truth the estimator is checked against, and where one
/// exists it is reported instead of a formula rather than averaged with it.
/// Mixing a measurement into an estimate produces a number that is neither.
#[derive(Clone, Copy, Debug)]
pub struct Measurement {
    /// Decode throughput, tokens/second x10.
    pub tps_x10: u16,
    /// Time to first token, milliseconds. Zero means it was not recorded.
    pub ttft_ms: u16,
    /// The quantization that was actually run, if it could be determined.
    pub quant: Option<Quant>,
    /// How many submitted runs were aggregated into this figure.
    pub runs: u8,
    pub provider: Provider,
}

impl Measurement {
    pub const fn tps(&self) -> f32 {
        self.tps_x10 as f32 / 10.0
    }
}

/// What a model is for. Drives the scoring weights.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum UseCase {
    General = 0,
    Coding = 1,
    Reasoning = 2,
    Chat = 3,
    Multimodal = 4,
    Embedding = 5,
}

impl UseCase {
    pub const fn name(self) -> &'static str {
        match self {
            UseCase::General => "general",
            UseCase::Coding => "coding",
            UseCase::Reasoning => "reasoning",
            UseCase::Chat => "chat",
            UseCase::Multimodal => "multimodal",
            UseCase::Embedding => "embedding",
        }
    }

    pub const fn from_u8(v: u8) -> UseCase {
        match v {
            1 => UseCase::Coding,
            2 => UseCase::Reasoning,
            3 => UseCase::Chat,
            4 => UseCase::Multimodal,
            5 => UseCase::Embedding,
            _ => UseCase::General,
        }
    }

    pub fn parse(s: &str) -> Option<UseCase> {
        match s {
            "general" => Some(UseCase::General),
            "coding" | "code" => Some(UseCase::Coding),
            "reasoning" | "reason" => Some(UseCase::Reasoning),
            "chat" => Some(UseCase::Chat),
            "multimodal" | "vision" => Some(UseCase::Multimodal),
            "embedding" | "embed" => Some(UseCase::Embedding),
            _ => None,
        }
    }
}

pub mod flags {
    /// Mixture-of-experts: only `active_params_m` move per token.
    pub const MOE: u16 = 1 << 0;
    /// Accepts images.
    pub const VISION: u16 = 1 << 1;
    /// Embedding model, not a generator.
    pub const EMBEDDING: u16 = 1 << 2;
    /// hidden/layers/kv_heads/head_dim are real values from the model config
    /// rather than estimates. Without this the KV-cache figure is inferred.
    pub const ARCH_METADATA: u16 = 1 << 3;
    /// Instruction-tuned rather than a base model.
    pub const INSTRUCT: u16 = 1 << 4;
    /// A GGUF conversion is known to exist, so the quant hierarchy is real.
    pub const GGUF: u16 = 1 << 5;
}

/// A parsed catalog borrowing its backing bytes.
#[derive(Clone, Copy)]
pub struct Catalog<'a> {
    records: &'a [u8],
    strings: &'a [u8],
    /// Measurements, sorted by `(hw_key, model_index)` so a lookup is a binary
    /// search rather than a scan. Empty when the catalog carries no benchmarks.
    measurements: &'a [u8],
    /// Per-hardware correction factors, sorted by `hw_key`.
    calibrations: &'a [u8],
    count: u32,
}

impl<'a> Catalog<'a> {
    /// Validate the header and bind the record and string sections.
    ///
    /// Every offset is bounds-checked here, once, so the accessors can decode
    /// without returning `Result` on every field.
    pub fn parse(data: &'a [u8]) -> Result<Catalog<'a>, CatalogError> {
        if data.len() < HEADER_LEN {
            return Err(CatalogError::TooShort);
        }
        if data[0..4] != MAGIC {
            return Err(CatalogError::BadMagic);
        }
        let version = u16le(data, 4);
        if version != FORMAT_VERSION {
            return Err(CatalogError::UnsupportedVersion(version));
        }
        let rec_size = u16le(data, 6);
        if rec_size as usize != RECORD_LEN {
            return Err(CatalogError::BadRecordSize(rec_size));
        }
        let count = u32le(data, 8);
        let rec_off = u32le(data, 12) as usize;
        let str_off = u32le(data, 16) as usize;
        let str_len = u32le(data, 20) as usize;
        let meas_off = u32le(data, 24) as usize;
        let meas_count = u32le(data, 28) as usize;
        let cal_off = u32le(data, 32) as usize;
        let cal_count = u32le(data, 36) as usize;

        let records = section(data, rec_off, count as usize, RECORD_LEN)?;
        let strings = section(data, str_off, str_len, 1)?;
        let measurements = section(data, meas_off, meas_count, MEASUREMENT_LEN)?;
        let calibrations = section(data, cal_off, cal_count, CALIBRATION_LEN)?;

        Ok(Catalog {
            records,
            strings,
            measurements,
            calibrations,
            count,
        })
    }

    /// How many real measurements this catalog carries.
    pub const fn measurement_count(&self) -> usize {
        self.measurements.len() / MEASUREMENT_LEN
    }

    /// How many machines have a calibration factor.
    pub const fn calibration_count(&self) -> usize {
        self.calibrations.len() / CALIBRATION_LEN
    }

    /// A measurement of this exact model on this exact machine, if one exists.
    ///
    /// Binary search over a section sorted by `(hw_key, model_index)`: about
    /// ten comparisons against a thousand measurements, with no allocation and
    /// no scan, which is what makes this affordable inside a ranking loop over
    /// the whole catalog.
    pub fn measurement(&self, hw_key: u32, model_index: u32) -> Option<Measurement> {
        if hw_key == 0 || self.measurements.is_empty() {
            return None;
        }
        let needle = ((hw_key as u64) << 32) | model_index as u64;
        let (mut lo, mut hi) = (0usize, self.measurement_count());
        while lo < hi {
            let mid = (lo + hi) / 2;
            let at = mid * MEASUREMENT_LEN;
            let key = ((u32le(self.measurements, at) as u64) << 32)
                | u32le(self.measurements, at + 4) as u64;
            match key.cmp(&needle) {
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
                core::cmp::Ordering::Equal => {
                    return Some(Measurement {
                        tps_x10: u16le(self.measurements, at + 8),
                        ttft_ms: u16le(self.measurements, at + 10),
                        quant: Quant::from_u8(self.measurements[at + 12]),
                        runs: self.measurements[at + 13],
                        provider: Provider::from_u8(self.measurements[at + 14]),
                    });
                }
            }
        }
        None
    }

    /// The correction factor derived for this machine, if anyone has ever
    /// benchmarked anything on it.
    ///
    /// This is what makes a handful of submissions useful far beyond the
    /// models they covered: if the formula ran 30% optimistic across every
    /// model measured on some card, it is running about 30% optimistic on the
    /// rest of the catalog too. Returned as a multiplier on the roofline
    /// estimate, along with how many measurements it was derived from so a
    /// caller can weigh it.
    pub fn calibration(&self, hw_key: u32) -> Option<(f32, u8)> {
        if hw_key == 0 || self.calibrations.is_empty() {
            return None;
        }
        let (mut lo, mut hi) = (0usize, self.calibration_count());
        while lo < hi {
            let mid = (lo + hi) / 2;
            let at = mid * CALIBRATION_LEN;
            let key = u32le(self.calibrations, at);
            match key.cmp(&hw_key) {
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
                core::cmp::Ordering::Equal => {
                    let factor = u16le(self.calibrations, at + 4) as f32 / 1000.0;
                    return Some((factor, self.calibrations[at + 6]));
                }
            }
        }
        None
    }

    pub const fn len(&self) -> usize {
        self.count as usize
    }

    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn get(&self, index: usize) -> Option<Model<'a>> {
        if index >= self.count as usize {
            return None;
        }
        let at = index * RECORD_LEN;
        Some(Model {
            raw: &self.records[at..at + RECORD_LEN],
            strings: self.strings,
            index: index as u32,
        })
    }

    pub fn iter(&self) -> ModelIter<'a> {
        ModelIter {
            catalog: *self,
            next: 0,
        }
    }

    /// Case-insensitive substring search over model names. Returns the first
    /// exact name match if there is one, otherwise the first substring hit,
    /// so `find("mistral-7b")` does not get shadowed by a longer name that
    /// happens to sort earlier.
    pub fn find(&self, needle: &str) -> Option<Model<'a>> {
        let mut first_partial: Option<Model<'a>> = None;
        for model in self.iter() {
            let name = model.name();
            if eq_ignore_case(name, needle) {
                return Some(model);
            }
            if first_partial.is_none() && contains_ignore_case(name, needle) {
                first_partial = Some(model);
            }
        }
        first_partial
    }
}

pub struct ModelIter<'a> {
    catalog: Catalog<'a>,
    next: usize,
}

impl<'a> Iterator for ModelIter<'a> {
    type Item = Model<'a>;

    fn next(&mut self) -> Option<Model<'a>> {
        let item = self.catalog.get(self.next)?;
        self.next += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.catalog.len().saturating_sub(self.next);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ModelIter<'_> {}

/// One catalog entry, decoded on access from its 32 backing bytes.
#[derive(Clone, Copy)]
pub struct Model<'a> {
    raw: &'a [u8],
    strings: &'a [u8],
    index: u32,
}

impl<'a> Model<'a> {
    /// Position in the catalog. Measurements are keyed by this rather than by
    /// a name hash, so matching a benchmark to a model happens once, in the
    /// packer, where the fuzzy name matching can be inspected and corrected —
    /// not on every device, every time, with no way to tell it went wrong.
    pub const fn index(&self) -> u32 {
        self.index
    }

    pub fn name(&self) -> &'a str {
        let off = u32le(self.raw, 0) as usize;
        read_cstr(self.strings, off)
    }

    /// Total parameters in millions. Zero means the catalog did not know.
    pub fn params_m(&self) -> u32 {
        u32le(self.raw, 4)
    }

    /// Parameters actually read per decoded token. Equals `params_m` for dense
    /// models; for MoE it is the router plus the active experts, which is what
    /// bandwidth-bound decode speed depends on.
    pub fn active_params_m(&self) -> u32 {
        let active = u32le(self.raw, 8);
        if active == 0 {
            self.params_m()
        } else {
            active
        }
    }

    /// Native context window in tokens.
    pub fn context_length(&self) -> u32 {
        u32le(self.raw, 12)
    }

    pub fn hidden_size(&self) -> u16 {
        u16le(self.raw, 16)
    }

    pub fn num_layers(&self) -> u16 {
        u16le(self.raw, 18)
    }

    pub fn num_kv_heads(&self) -> u16 {
        u16le(self.raw, 20)
    }

    pub fn head_dim(&self) -> u16 {
        u16le(self.raw, 22)
    }

    /// Vocabulary size in thousands of tokens.
    pub fn vocab_k(&self) -> u16 {
        u16le(self.raw, 24)
    }

    pub fn flags(&self) -> u16 {
        u16le(self.raw, 26)
    }

    pub fn has_flag(&self, flag: u16) -> bool {
        self.flags() & flag != 0
    }

    pub fn is_moe(&self) -> bool {
        self.has_flag(flags::MOE)
    }

    /// True when layer and head counts came from the model config rather than
    /// from the size-class estimator. Callers surface this so a KV-cache
    /// figure is never mistaken for a measured one.
    pub fn has_arch_metadata(&self) -> bool {
        self.has_flag(flags::ARCH_METADATA)
    }

    /// Bitmask of quantizations known to be published for this model.
    /// Bit N corresponds to `Quant` discriminant N.
    pub fn quant_mask(&self) -> u8 {
        self.raw[28]
    }

    pub fn supports_quant(&self, q: Quant) -> bool {
        let mask = self.quant_mask();
        // An empty mask means the packer had no per-file information. Assume
        // the standard GGUF ladder rather than declaring the model unusable.
        mask == 0 || mask & (1u8 << (q as u8)) != 0
    }

    pub fn use_case(&self) -> UseCase {
        UseCase::from_u8(self.raw[29])
    }

    pub fn family_id(&self) -> u8 {
        self.raw[30]
    }

    /// Curated quality prior, 0-255, folded into the quality score alongside
    /// parameter count. Lets a well-regarded 7B outrank a mediocre 13B.
    pub fn quality_prior(&self) -> u8 {
        self.raw[31]
    }

    /// Parameters in billions, as a float, for the estimator.
    pub fn params_b(&self) -> f32 {
        self.params_m() as f32 / 1000.0
    }

    pub fn active_params_b(&self) -> f32 {
        self.active_params_m() as f32 / 1000.0
    }
}

// --- decoding helpers -------------------------------------------------------

/// Bind one section, refusing anything that would read past the buffer.
///
/// All the bounds checking happens here, once, at parse time. That is what
/// lets every accessor below index without returning a `Result`, and it means
/// a truncated or corrupt catalog is rejected at the door rather than causing
/// a panic on a device with no way to report one.
fn section(data: &[u8], offset: usize, count: usize, stride: usize) -> Result<&[u8], CatalogError> {
    if count == 0 {
        return Ok(&[]);
    }
    let len = count.checked_mul(stride).ok_or(CatalogError::Truncated)?;
    let end = offset.checked_add(len).ok_or(CatalogError::Truncated)?;
    if end > data.len() {
        return Err(CatalogError::Truncated);
    }
    Ok(&data[offset..end])
}

#[inline]
fn u16le(buf: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([buf[at], buf[at + 1]])
}

#[inline]
fn u32le(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]])
}

/// Read a NUL-terminated UTF-8 string from the string table.
///
/// A malformed offset yields an empty name rather than a panic: a corrupt
/// catalog should degrade to useless output, not take down a device that has
/// no way to report a backtrace.
fn read_cstr(strings: &[u8], off: usize) -> &str {
    if off >= strings.len() {
        return "";
    }
    let rest = &strings[off..];
    let end = match rest.iter().position(|&b| b == 0) {
        Some(p) => p,
        None => rest.len(),
    };
    core::str::from_utf8(&rest[..end]).unwrap_or("")
}

/// Suffixes that mark a repackage rather than a different model.
///
/// The catalog is full of the same weights republished as GGUF, AWQ, MLX, or
/// a 4-bit bitsandbytes dump. They are genuinely useful downloads but they are
/// not distinct recommendations, and left alone they fill a top-10 list with
/// eight spellings of one model.
const REPACKAGE_SUFFIXES: &[&str] = &[
    "-gguf",
    "-awq",
    "-gptq",
    "-exl2",
    "-mlx",
    "-mxfp4-q8",
    "-mxfp4",
    "-bf16",
    "-fp16",
    "-fp8",
    "-int4",
    "-int8",
    "-4bit",
    "-8bit",
    "-q8",
    "-q4",
    "-bnb",
    "-unsloth",
    "-quantized",
    "-hf",
];

/// The part of a model name that identifies the underlying weights.
///
/// Drops the publishing org and any repackaging suffix, so
/// `unsloth/gpt-oss-20b-BF16` and `openai/gpt-oss-20b` collapse to the same
/// key. Returns a subslice, never an allocation, because this runs inside the
/// ranking loop on a device with no allocator.
pub fn base_name(name: &str) -> &str {
    let after_org = match name.rfind('/') {
        Some(i) => &name[i + 1..],
        None => name,
    };
    let mut out = after_org;
    // Suffixes stack — `-unsloth-bnb-4bit` is really three — so strip until
    // nothing more comes off.
    loop {
        let mut stripped = false;
        for suffix in REPACKAGE_SUFFIXES {
            let cut = out.len().saturating_sub(suffix.len());
            if cut > 0 && out.is_char_boundary(cut) && ends_with_ignore_case(out, suffix) {
                out = &out[..cut];
                stripped = true;
            }
        }
        if !stripped {
            break;
        }
    }
    out
}

/// Whether two names denote the same underlying weights.
pub fn same_weights(a: &str, b: &str) -> bool {
    eq_ignore_case(base_name(a), base_name(b))
}

fn ends_with_ignore_case(hay: &str, suffix: &str) -> bool {
    let (h, s) = (hay.as_bytes(), suffix.as_bytes());
    if s.len() > h.len() {
        return false;
    }
    h[h.len() - s.len()..].eq_ignore_ascii_case(s)
}

fn eq_ignore_case(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

fn contains_ignore_case(hay: &str, needle: &str) -> bool {
    let (h, n) = (hay.as_bytes(), needle.as_bytes());
    if n.is_empty() {
        return true;
    }
    if n.len() > h.len() {
        return false;
    }
    for start in 0..=(h.len() - n.len()) {
        if h[start..start + n.len()].eq_ignore_ascii_case(n) {
            return true;
        }
    }
    false
}
