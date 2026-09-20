//! Ingesting community benchmark submissions.
//!
//! Real measurements are the only thing in this project that is not arithmetic,
//! and they are worth a great deal: a number somebody recorded on the hardware
//! in front of them beats any formula. The cost is that submissions name models
//! however the submitter's runtime named them — `qwen3:8b` from Ollama,
//! `Qwen3-8B-Q4_K_M.gguf` from llama.cpp, an MLX repo path — and none of those
//! is the catalog's name for the model.
//!
//! All of that reconciliation happens here, at pack time, where it can be
//! counted and inspected. What ships is a measurement keyed by catalog index,
//! so the device does no string matching at all and cannot get it wrong in a
//! way nobody can see.

use std::collections::HashMap;
use std::path::Path;

use openchoice_core::catalog::hw_key;
use openchoice_core::{Backend, Hardware, Quant};

/// One submitted benchmark file.
#[derive(serde::Deserialize)]
pub struct Submission {
    pub hardware: SubmittedHardware,
    #[serde(default)]
    pub results: Vec<SubmittedResult>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubmittedHardware {
    #[serde(default)]
    pub hardware_name: Option<String>,
    #[serde(default)]
    pub hw_class: Option<String>,
    #[serde(default)]
    pub ram_gb: Option<f64>,
    #[serde(default)]
    pub vram_gb: Option<f64>,
    #[serde(default)]
    pub unified_memory: Option<bool>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubmittedResult {
    pub model: String,
    #[serde(default)]
    pub avg_tps: Option<f64>,
    #[serde(default)]
    pub avg_ttft_ms: Option<f64>,
    #[serde(default)]
    pub num_runs: Option<u32>,
    #[serde(default)]
    pub provider: Option<String>,
}

/// Tokens that describe how a model was packaged or tuned, not which model it
/// is. Two submissions of the same weights at different quantizations are
/// measurements of the same catalog entry, so these are dropped before
/// matching. The quantization itself is recovered separately and kept.
const DROP_TOKENS: &[&str] = &[
    // quantization and packaging
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q8",
    "k",
    "m",
    "s",
    "l",
    "f16",
    "fp16",
    "bf16",
    "f32",
    "mxfp4",
    "mxfp8",
    "nvfp4",
    "int4",
    "int8",
    "fp8",
    "4bit",
    "8bit",
    "gguf",
    "awq",
    "gptq",
    "exl2",
    "mlx",
    "bnb",
    "unsloth",
    "qat",
    "ud",
    "xl",
    "quantized",
    // tuning and serving labels: a base and an instruct tune of one model
    // decode at the same speed, which is all a measurement records
    "instruct",
    "it",
    "chat",
    "sft",
    "dpo",
    "hf",
    "base",
    "text",
    "abliterated",
    "uncensored",
    "preview",
    "latest",
    "pt",
    "distill",
];

/// Tokens that mark the start of a submitter's own serving configuration —
/// `-agent-gpu50-latest`, `-enfixed`, `-v2ctx65k`. Everything from here on
/// describes their setup, not a published model, so the name is truncated.
const TAIL_MARKERS: &[&str] = &["agent", "enfixed", "cpu"];

/// Reduce a model name from any source to a comparable key.
///
/// Deliberately lossy: it collapses quantization, fine-tune labels, and
/// packaging into a family-plus-size identity, because that is the granularity
/// at which decode throughput is actually determined.
pub fn normalize_model(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    let base = lower.rsplit('/').next().unwrap_or(&lower);
    let base = base.strip_suffix(".gguf").unwrap_or(base);
    let base = base.replace(':', "-");

    let mut tokens: Vec<String> = Vec::new();
    for raw in base.split(['-', '_', '.', ' ']) {
        if raw.is_empty() {
            continue;
        }
        if TAIL_MARKERS.contains(&raw) || is_gpu_tag(raw) || is_ctx_tag(raw) {
            break;
        }
        // "gemma4" and "llama3" are the same models the catalog spells
        // "gemma-4" and "llama-3". Split a family glued to its version so the
        // two spellings meet. A size token like "4b" is left alone.
        match split_family_version(raw) {
            Some((family, version)) => {
                tokens.push(family.to_string());
                tokens.push(version.to_string());
            }
            None => tokens.push(raw.to_string()),
        }
    }

    let mut kept: Vec<String> = tokens
        .into_iter()
        .filter(|t| !DROP_TOKENS.contains(&t.as_str()))
        .collect();

    // `Meta-Llama-3.1-8B` is the catalog's `Llama-3.1-8B`.
    if kept.len() > 1 && kept[0] == "meta" && kept[1] == "llama" {
        kept.remove(0);
    }
    // Trailing release dates (`2507`, `2410`) distinguish refreshes that run
    // at identical speed.
    while kept
        .last()
        .is_some_and(|t| t.len() == 4 && t.chars().all(|c| c.is_ascii_digit()))
    {
        kept.pop();
    }

    kept.join("-")
}

fn split_family_version(token: &str) -> Option<(&str, &str)> {
    let split = token.find(|c: char| c.is_ascii_digit())?;
    if split < 3 {
        return None;
    }
    let (family, version) = token.split_at(split);
    if !family.chars().all(|c| c.is_ascii_alphabetic())
        || !version.chars().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    Some((family, version))
}

fn is_gpu_tag(token: &str) -> bool {
    token
        .strip_prefix("gpu")
        .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
}

fn is_ctx_tag(token: &str) -> bool {
    token.contains("ctx") && token.chars().any(|c| c.is_ascii_digit())
}

/// Recover the quantization a benchmark actually ran at, when the name says.
pub fn quant_from_name(name: &str) -> Option<Quant> {
    let lower = name.to_ascii_lowercase();
    for (needle, q) in [
        ("q4_k_m", Quant::Q4KM),
        ("q4_k_s", Quant::Q4KM),
        ("q4_0", Quant::Q4KM),
        ("q4km", Quant::Q4KM),
        ("q5_k_m", Quant::Q5KM),
        ("q5_k", Quant::Q5KM),
        ("q6_k", Quant::Q6K),
        ("q8_0", Quant::Q8_0),
        ("q3_k", Quant::Q3KM),
        ("q2_k", Quant::Q2K),
        ("mxfp4", Quant::Q4KM),
        ("int4", Quant::Q4KM),
        ("4bit", Quant::Q4KM),
        ("8bit", Quant::Q8_0),
        ("bf16", Quant::F16),
        ("fp16", Quant::F16),
        ("f16", Quant::F16),
    ] {
        if lower.contains(needle) {
            return Some(q);
        }
    }
    // Ollama, llama.cpp and LM Studio all default to Q4_K_M, and a submission
    // whose name carries no quantization tag is overwhelmingly one of those
    // default pulls. Recording that rather than "unknown" is what lets the
    // engine rescale the measurement onto a different quantization instead of
    // discarding it — and being wrong here costs a rescale factor, not a
    // fabricated measurement.
    Some(Quant::Q4KM)
}

pub fn provider_code(name: Option<&str>) -> u8 {
    match name.unwrap_or("").to_ascii_lowercase().as_str() {
        "llamacpp" | "llama.cpp" => 1,
        "ollama" => 2,
        "mlx" => 3,
        "vllm" => 4,
        _ => 0,
    }
}

/// A submitted machine, turned into something the engine can score against so
/// the formula's prediction for it can be compared with what was measured.
pub struct Machine {
    pub key: u32,
    pub name: String,
    pub hardware: Hardware,
}

pub fn machine_from(h: &SubmittedHardware) -> Option<Machine> {
    let name = h.hardware_name.as_deref()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let ram_mb = (h.ram_gb.unwrap_or(0.0) * 1024.0) as u32;
    if ram_mb == 0 {
        return None;
    }
    let unified = h.unified_memory.unwrap_or(false) || h.hw_class.as_deref() == Some("UNIFIED");
    let vram_mb = (h.vram_gb.unwrap_or(0.0) * 1024.0) as u32;

    let lower = name.to_ascii_lowercase();
    let backend = if lower.contains("apple") || lower.starts_with('m') && unified {
        Backend::Metal
    } else if lower.contains("nvidia") || lower.contains("geforce") || lower.contains("rtx") {
        Backend::Cuda
    } else if lower.contains("amd") || lower.contains("radeon") {
        Backend::Rocm
    } else if lower.contains("intel") || lower.contains("arc") {
        Backend::Sycl
    } else {
        Backend::CpuX86
    };

    let (bw, tflops) = openchoice_core::lookup_gpu(&name).unwrap_or((0, 0));

    Some(Machine {
        key: hw_key(&name),
        name,
        hardware: Hardware {
            ram_mb,
            vram_mb,
            unified,
            backend,
            gpu_bandwidth_gbps: bw,
            ram_bandwidth_gbps: 0,
            tflops_fp16_x10: tflops,
            os_reserve_mb: 2048,
            hw_key: 0, // not used for the prediction side
        },
    })
}

/// One aggregated measurement ready to pack.
pub struct PackedMeasurement {
    pub hw_key: u32,
    pub model_index: u32,
    pub tps_x10: u16,
    pub ttft_ms: u16,
    pub quant: u8,
    pub runs: u8,
    pub provider: u8,
}

/// What ingestion produced, including what it failed to match — an import that
/// silently drops half its input is worse than one that says so.
pub struct Ingested {
    pub measurements: Vec<PackedMeasurement>,
    pub calibrations: Vec<(u32, u16, u8)>,
    pub files: usize,
    pub results: usize,
    pub matched: usize,
    pub machines: usize,
    pub unmatched: Vec<(String, usize)>,
}

struct Sample {
    tps: f64,
    ttft: f64,
    runs: u32,
    quant: u8,
    provider: u8,
}

pub fn ingest(
    dir: &Path,
    index_by_key: &HashMap<String, u32>,
    predict: impl Fn(&Hardware, u32, Option<Quant>) -> Option<f64>,
) -> Result<Ingested, String> {
    let mut files = 0usize;
    let mut results = 0usize;
    let mut matched = 0usize;
    let mut unmatched: HashMap<String, usize> = HashMap::new();
    let mut machines: HashMap<u32, String> = HashMap::new();
    // (hw_key, model_index) -> samples
    let mut grouped: HashMap<(u32, u32), Vec<Sample>> = HashMap::new();
    // hw_key -> (hardware, ratios)
    let mut ratios: HashMap<u32, Vec<f64>> = HashMap::new();

    for entry in walk(dir)? {
        let text = match std::fs::read_to_string(&entry) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let sub: Submission = match serde_json::from_str(&text) {
            Ok(s) => s,
            // A malformed submission is skipped, not fatal: one bad file in a
            // community directory must not stop the catalog from building.
            Err(_) => continue,
        };
        files += 1;

        let Some(machine) = machine_from(&sub.hardware) else {
            continue;
        };
        machines.insert(machine.key, machine.name.clone());

        for r in &sub.results {
            results += 1;
            let Some(tps) = r.avg_tps.filter(|v| *v > 0.0) else {
                continue;
            };
            let key = normalize_model(&r.model);
            let Some(&model_index) = index_by_key.get(&key) else {
                *unmatched.entry(key).or_default() += 1;
                continue;
            };
            matched += 1;

            let quant = quant_from_name(&r.model);
            grouped
                .entry((machine.key, model_index))
                .or_default()
                .push(Sample {
                    tps,
                    ttft: r.avg_ttft_ms.unwrap_or(0.0),
                    runs: r.num_runs.unwrap_or(1),
                    quant: quant.map(|q| q as u8).unwrap_or(0xFF),
                    provider: provider_code(r.provider.as_deref()),
                });

            // How far off was the formula for this pairing on this machine?
            if let Some(predicted) = predict(&machine.hardware, model_index, quant) {
                if predicted > 0.0 {
                    let ratio = tps / predicted;
                    // Ratios outside this band mean the pairing was
                    // misidentified, not that the formula is off by 20x.
                    // Feeding those into a correction factor would poison it.
                    if (0.05..=5.0).contains(&ratio) {
                        ratios.entry(machine.key).or_default().push(ratio);
                    }
                }
            }
        }
    }

    let mut measurements: Vec<PackedMeasurement> = grouped
        .into_iter()
        .map(|((hw, model_index), mut samples)| {
            // Median, not mean: a single thermally-throttled or
            // background-loaded run should not drag the figure.
            samples.sort_by(|a, b| a.tps.partial_cmp(&b.tps).unwrap());
            let mid = samples.len() / 2;
            let tps = samples[mid].tps;
            let ttfts: Vec<f64> = samples
                .iter()
                .map(|s| s.ttft)
                .filter(|v| *v > 0.0)
                .collect();
            let ttft = if ttfts.is_empty() {
                0.0
            } else {
                ttfts[ttfts.len() / 2]
            };
            PackedMeasurement {
                hw_key: hw,
                model_index,
                tps_x10: (tps * 10.0).round().min(u16::MAX as f64) as u16,
                ttft_ms: ttft.round().min(u16::MAX as f64) as u16,
                quant: samples[mid].quant,
                runs: samples
                    .iter()
                    .map(|s| s.runs)
                    .sum::<u32>()
                    .min(u8::MAX as u32) as u8,
                provider: samples[mid].provider,
            }
        })
        .collect();
    measurements.sort_by_key(|m| ((m.hw_key as u64) << 32) | m.model_index as u64);

    let mut calibrations: Vec<(u32, u16, u8)> = ratios
        .into_iter()
        .filter(|(_, r)| r.len() >= 2) // one sample is an anecdote, not a factor
        .map(|(hw, mut r)| {
            r.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = r[r.len() / 2];
            (
                hw,
                (median * 1000.0).round().clamp(50.0, 5000.0) as u16,
                r.len().min(u8::MAX as usize) as u8,
            )
        })
        .collect();
    calibrations.sort_by_key(|c| c.0);

    let mut unmatched: Vec<(String, usize)> = unmatched.into_iter().collect();
    unmatched.sort_by_key(|(_, count)| core::cmp::Reverse(*count));

    Ok(Ingested {
        measurements,
        calibrations,
        files,
        results,
        matched,
        machines: machines.len(),
        unmatched,
    })
}

fn walk(dir: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| format!("reading {}: {e}", dir.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk(&path)?);
        } else if path.extension().is_some_and(|e| e == "json") {
            out.push(path);
        }
    }
    Ok(out)
}
