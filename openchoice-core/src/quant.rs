//! Quantization formats and the hierarchy the engine walks.
//!
//! Bits-per-weight figures are the effective whole-file rates observed in
//! llama.cpp GGUF conversions, not the nominal block widths: a "4-bit" K-quant
//! stores scales and mins alongside the nibbles and lands near 4.8 bpw in
//! practice. Using nominal widths here under-counts weights by ~20%, which is
//! the difference between "fits" and "OOM on load" for a tight pairing.

/// A quantization format, ordered best-quality first.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum Quant {
    F16 = 0,
    Q8_0 = 1,
    Q6K = 2,
    Q5KM = 3,
    Q4KM = 4,
    Q3KM = 5,
    Q2K = 6,
}

impl Quant {
    /// Best quality first. The engine walks this in order and takes the first
    /// entry that fits, so the ordering is the policy.
    pub const HIERARCHY: [Quant; 7] = [
        Quant::F16,
        Quant::Q8_0,
        Quant::Q6K,
        Quant::Q5KM,
        Quant::Q4KM,
        Quant::Q3KM,
        Quant::Q2K,
    ];

    /// Effective bits per weight across the whole file, including scale and
    /// min metadata carried by the K-quant block formats.
    pub const fn bits_per_weight(self) -> f32 {
        match self {
            Quant::F16 => 16.00,
            Quant::Q8_0 => 8.50,
            Quant::Q6K => 6.56,
            Quant::Q5KM => 5.67,
            Quant::Q4KM => 4.83,
            Quant::Q3KM => 3.91,
            Quant::Q2K => 3.35,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Quant::F16 => "F16",
            Quant::Q8_0 => "Q8_0",
            Quant::Q6K => "Q6_K",
            Quant::Q5KM => "Q5_K_M",
            Quant::Q4KM => "Q4_K_M",
            Quant::Q3KM => "Q3_K_M",
            Quant::Q2K => "Q2_K",
        }
    }

    /// Fraction of the fp16 model's output quality retained, as a multiplier
    /// on the quality score. Derived from perplexity deltas published in the
    /// llama.cpp quantization comparisons; the cliff below Q4 is real and the
    /// curve is deliberately steep there.
    pub const fn quality_factor(self) -> f32 {
        match self {
            Quant::F16 => 1.000,
            Quant::Q8_0 => 0.998,
            Quant::Q6K => 0.995,
            Quant::Q5KM => 0.988,
            Quant::Q4KM => 0.972,
            Quant::Q3KM => 0.922,
            Quant::Q2K => 0.810,
        }
    }

    /// Decode speed relative to Q4_K_M at equal model size. Lower-bit formats
    /// move fewer bytes per token but pay more dequantization ALU, so the
    /// curve is not simply proportional to bpw.
    pub const fn speed_multiplier(self) -> f32 {
        match self {
            Quant::F16 => 0.52,
            Quant::Q8_0 => 0.78,
            Quant::Q6K => 0.90,
            Quant::Q5KM => 0.96,
            Quant::Q4KM => 1.00,
            Quant::Q3KM => 1.04,
            Quant::Q2K => 1.08,
        }
    }

    pub const fn from_u8(v: u8) -> Option<Quant> {
        match v {
            0 => Some(Quant::F16),
            1 => Some(Quant::Q8_0),
            2 => Some(Quant::Q6K),
            3 => Some(Quant::Q5KM),
            4 => Some(Quant::Q4KM),
            5 => Some(Quant::Q3KM),
            6 => Some(Quant::Q2K),
            _ => None,
        }
    }

    /// Parse the spellings that appear in GGUF filenames and model cards.
    pub fn parse(s: &str) -> Option<Quant> {
        let mut buf = [0u8; 16];
        let bytes = s.as_bytes();
        let n = if bytes.len() < 16 { bytes.len() } else { 16 };
        for i in 0..n {
            buf[i] = bytes[i].to_ascii_uppercase();
        }
        let up = core::str::from_utf8(&buf[..n]).ok()?;
        match up {
            "F16" | "FP16" | "BF16" | "F32" => Some(Quant::F16),
            "Q8_0" | "Q8" => Some(Quant::Q8_0),
            "Q6_K" | "Q6K" | "Q6" => Some(Quant::Q6K),
            "Q5_K_M" | "Q5_K_S" | "Q5_K" | "Q5_0" | "Q5" => Some(Quant::Q5KM),
            "Q4_K_M" | "Q4_K_S" | "Q4_K" | "Q4_0" | "Q4" => Some(Quant::Q4KM),
            "Q3_K_M" | "Q3_K_S" | "Q3_K_L" | "Q3_K" | "Q3" => Some(Quant::Q3KM),
            "Q2_K" | "Q2K" | "Q2" => Some(Quant::Q2K),
            _ => None,
        }
    }
}

/// KV-cache element width. Halving the cache is often what turns a Too Tight
/// long-context pairing into a usable one, so it is a first-class knob rather
/// than a hidden constant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum KvQuant {
    F16 = 0,
    Q8 = 1,
    Q4 = 2,
}

impl KvQuant {
    pub const fn bytes_per_element(self) -> f32 {
        match self {
            KvQuant::F16 => 2.0,
            KvQuant::Q8 => 1.0,
            KvQuant::Q4 => 0.5,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            KvQuant::F16 => "f16",
            KvQuant::Q8 => "q8_0",
            KvQuant::Q4 => "q4_0",
        }
    }
}
