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

/// `"OCB1"` — OpenChoice Binary catalog, version 1.
pub const MAGIC: [u8; 4] = *b"OCB1";
pub const FORMAT_VERSION: u16 = 1;
pub const HEADER_LEN: usize = 32;
pub const RECORD_LEN: usize = 32;

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

        let rec_len = (count as usize)
            .checked_mul(RECORD_LEN)
            .ok_or(CatalogError::Truncated)?;
        let rec_end = rec_off
            .checked_add(rec_len)
            .ok_or(CatalogError::Truncated)?;
        let str_end = str_off
            .checked_add(str_len)
            .ok_or(CatalogError::Truncated)?;
        if rec_end > data.len() || str_end > data.len() {
            return Err(CatalogError::Truncated);
        }

        Ok(Catalog {
            records: &data[rec_off..rec_end],
            strings: &data[str_off..str_end],
            count,
        })
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
}

impl<'a> Model<'a> {
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
