//! Packs a JSON model catalog into the `.ocb` format `openchoice-core` reads
//! in place.
//!
//! The input schema is the one the upstream llmfit scraper emits, so an
//! existing `hf_models.json` can be packed without conversion. Output is a
//! header, a run of fixed-width 32-byte records, and a deduplicated string
//! table — around 1 MB for 15,000 models, against 13.4 MB of JSON.
//!
//! ```text
//! openchoice-catalog build --input hf_models.json --output catalog/openchoice.ocb
//! openchoice-catalog build --input hf_models.json --output tiny.ocb --top 512
//! openchoice-catalog inspect catalog/openchoice.ocb
//! ```

mod community;

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::process::ExitCode;

use openchoice_core::catalog::{
    flags, CALIBRATION_LEN, FORMAT_VERSION, HEADER_LEN, MAGIC, MEASUREMENT_LEN, RECORD_LEN,
};
use openchoice_core::{Catalog, Opts, Quant, UseCase};

/// One entry as it appears in the source JSON. Everything is optional because
/// the scraper cannot always resolve a model's config, and a missing field
/// must degrade the estimate rather than drop the model.
#[derive(Deserialize, Default)]
struct SourceModel {
    name: String,
    #[serde(default)]
    parameters_raw: Option<u64>,
    #[serde(default)]
    active_parameters: Option<u64>,
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default)]
    hidden_size: Option<u32>,
    #[serde(default)]
    num_hidden_layers: Option<u32>,
    #[serde(default)]
    num_key_value_heads: Option<u32>,
    #[serde(default)]
    head_dim: Option<u32>,
    #[serde(default)]
    vocab_size: Option<u32>,
    #[serde(default)]
    is_moe: Option<bool>,
    #[serde(default)]
    num_experts: Option<u32>,
    #[serde(default)]
    capabilities: Vec<String>,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default)]
    hf_downloads: Option<u64>,
    #[serde(default)]
    hf_likes: Option<u64>,
    #[serde(default)]
    gguf_sources: Option<serde_json::Value>,
}

/// Families we recognise by name, for grouping in the UI. Index into this is
/// the `family` byte; 0 means "not one of these".
const FAMILIES: &[&str] = &[
    "other",
    "llama",
    "qwen",
    "mistral",
    "gemma",
    "phi",
    "deepseek",
    "granite",
    "olmo",
    "falcon",
    "starcoder",
    "codellama",
    "yi",
    "command-r",
    "glm",
    "internlm",
    "minicpm",
    "stablelm",
    "bge",
    "nomic",
    "smollm",
    "tinyllama",
    "exaone",
    "kimi",
    "ernie",
    "grok",
    "nemotron",
    "hunyuan",
];

struct Args {
    command: String,
    input: Option<String>,
    output: Option<String>,
    top: Option<usize>,
    min_downloads: u64,
    include_all_pipelines: bool,
    community: Option<String>,
    show_unmatched: bool,
    path: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut raw = std::env::args().skip(1);
    let command = raw.next().unwrap_or_else(|| "help".into());
    let mut args = Args {
        command,
        input: None,
        output: None,
        top: None,
        min_downloads: 1000,
        include_all_pipelines: false,
        community: None,
        show_unmatched: false,
        path: None,
    };
    while let Some(flag) = raw.next() {
        match flag.as_str() {
            "--input" | "-i" => args.input = raw.next(),
            "--output" | "-o" => args.output = raw.next(),
            "--top" => {
                args.top = Some(
                    raw.next()
                        .ok_or("--top needs a value")?
                        .parse()
                        .map_err(|_| "--top must be a number")?,
                )
            }
            "--min-downloads" => {
                args.min_downloads = raw
                    .next()
                    .ok_or("--min-downloads needs a value")?
                    .parse()
                    .map_err(|_| "--min-downloads must be a number")?
            }
            "--all-pipelines" => args.include_all_pipelines = true,
            "--community" => args.community = raw.next(),
            "--show-unmatched" => args.show_unmatched = true,
            other if !other.starts_with('-') && args.path.is_none() => {
                args.path = Some(other.to_string())
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    Ok(args)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let result = match args.command.as_str() {
        "build" => build(&args),
        "inspect" => inspect(&args),
        _ => {
            usage();
            return ExitCode::SUCCESS;
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn usage() {
    eprintln!(
        "openchoice-catalog — pack a model catalog into .ocb\n\
         \n\
         USAGE:\n\
         \x20 openchoice-catalog build -i <models.json> -o <out.ocb> [--top N]\n\
         \x20                          [--min-downloads N] [--all-pipelines]\n\
         \x20 openchoice-catalog inspect <catalog.ocb>\n\
         \n\
         --top N          keep only the N most-downloaded models. A 512-entry\n\
         \x20                catalog is ~40 KB and fits anywhere.\n\
         --min-downloads  drop long-tail uploads below this download count.\n\
         --all-pipelines  keep audio, TTS, and other non-text models too."
    );
}

fn build(args: &Args) -> Result<(), String> {
    let input = args.input.as_deref().ok_or("build needs --input")?;
    let output = args.output.as_deref().ok_or("build needs --output")?;

    let raw = std::fs::read_to_string(input).map_err(|e| format!("reading {input}: {e}"))?;
    let models: Vec<SourceModel> =
        serde_json::from_str(&raw).map_err(|e| format!("parsing {input}: {e}"))?;
    let total = models.len();

    let mut kept: Vec<SourceModel> = models
        .into_iter()
        .filter(|m| keep(m, args.min_downloads, args.include_all_pipelines))
        .collect();

    // Most-downloaded first, so --top takes the models people actually use and
    // the catalog stays useful when truncated hard for a small flash budget.
    kept.sort_by(|a, b| {
        b.hf_downloads
            .unwrap_or(0)
            .cmp(&a.hf_downloads.unwrap_or(0))
            .then_with(|| a.name.cmp(&b.name))
    });
    if let Some(n) = args.top {
        kept.truncate(n);
    }

    // Pack once without measurements so the community importer has a real
    // catalog to resolve model names against and to predict throughput from.
    let base = pack(&kept, &[], &[])?;
    let ingested = match args.community.as_deref() {
        Some(dir) => Some(import_community(dir, &base, args.show_unmatched)?),
        None => None,
    };

    let (measurements, calibrations) = match &ingested {
        Some(i) => (i.measurements.as_slice(), i.calibrations.as_slice()),
        None => (&[][..], &[][..]),
    };
    let bytes = pack(&kept, measurements, calibrations)?;
    std::fs::write(output, &bytes).map_err(|e| format!("writing {output}: {e}"))?;

    // Round-trip immediately. A catalog that does not parse is worse than no
    // catalog, and this is the only place that can catch it before a device
    // tries to boot on it.
    let parsed = Catalog::parse(&bytes).map_err(|e| format!("output failed to re-parse: {e:?}"))?;
    if parsed.len() != kept.len() {
        return Err(format!(
            "round-trip mismatch: packed {} models, read back {}",
            kept.len(),
            parsed.len()
        ));
    }

    println!(
        "packed {} of {total} models into {output}\n  {} bytes ({:.1} KB), {} bytes/model\n  source JSON was {:.1} MB",
        kept.len(),
        bytes.len(),
        bytes.len() as f64 / 1024.0,
        bytes.len() / kept.len().max(1),
        raw.len() as f64 / 1_048_576.0
    );
    if let Some(i) = &ingested {
        println!(
            "  {} measurements from {} submissions — {} of {} results matched a catalog model",
            i.measurements.len(),
            i.files,
            i.matched,
            i.results
        );
        println!(
            "  {} machines seen, {} with enough samples for a calibration factor",
            i.machines,
            i.calibrations.len()
        );
    }
    Ok(())
}

/// Resolve community submissions against the catalog we just packed.
///
/// The prediction closure runs the real engine, so a calibration factor is the
/// ratio between what this code would have said and what actually happened —
/// not a ratio against some other formula that has since drifted.
fn import_community(
    dir: &str,
    base: &[u8],
    show_unmatched: bool,
) -> Result<community::Ingested, String> {
    let catalog = Catalog::parse(base).map_err(|e| format!("internal: {e:?}"))?;

    let mut index_by_key: HashMap<String, u32> = HashMap::new();
    for model in catalog.iter() {
        // Models are packed most-downloaded first, so the first entry to claim
        // a key is the canonical one and later repackages do not displace it.
        index_by_key
            .entry(community::normalize_model(model.name()))
            .or_insert(model.index());
    }

    let predict =
        |hw: &openchoice_core::Hardware, index: u32, quant: Option<Quant>| -> Option<f64> {
            let model = catalog.get(index as usize)?;
            let opts = Opts {
                // Benchmarks use short prompts; scoring at the model's full native
                // window would charge a KV cache the run never allocated.
                context: Some(4096),
                ..Default::default()
            };
            let fit = match quant {
                Some(q) => openchoice_core::fit::evaluate_at(&model, hw, &opts, q, 4096),
                None => openchoice_core::fit::evaluate(&model, hw, &opts),
            };
            let speed = openchoice_core::speed::estimate(&model, hw, &fit, opts.efficiency);
            Some(speed.decode_tps_x10 as f64 / 10.0)
        };

    let ingested = community::ingest(Path::new(dir), &index_by_key, predict)?;

    if show_unmatched {
        eprintln!(
            "
benchmark names that matched no catalog model:"
        );
        for (name, count) in ingested.unmatched.iter().take(40) {
            eprintln!("  {count:3}  {name}");
        }
    }
    Ok(ingested)
}

/// Whether a source entry earns a slot.
///
/// The upstream catalog carries a long tail of near-empty test uploads —
/// entries with two parameters and no downloads. They cost flash and can only
/// dilute a ranking, so they are dropped rather than packed.
fn keep(m: &SourceModel, min_downloads: u64, all_pipelines: bool) -> bool {
    if m.name.is_empty() {
        return false;
    }
    let params = m.parameters_raw.unwrap_or(0);
    if params < 1_000_000 {
        return false;
    }
    if m.hf_downloads.unwrap_or(0) < min_downloads {
        return false;
    }
    if !all_pipelines {
        let tag = m.pipeline_tag.as_deref().unwrap_or("text-generation");
        let usable = matches!(
            tag,
            "text-generation"
                | "text2text-generation"
                | "image-text-to-text"
                | "feature-extraction"
                | "sentence-similarity"
                | "fill-mask"
        );
        if !usable {
            return false;
        }
    }
    true
}

fn pack(
    models: &[SourceModel],
    measurements: &[community::PackedMeasurement],
    calibrations: &[(u32, u16, u8)],
) -> Result<Vec<u8>, String> {
    // Deduplicated string table. Model names repeat their org prefix often
    // enough that interning is worth the HashMap.
    let mut strings: Vec<u8> = Vec::new();
    let mut interned: HashMap<String, u32> = HashMap::new();

    fn intern(s: &str, table: &mut Vec<u8>, map: &mut HashMap<String, u32>) -> u32 {
        if let Some(&off) = map.get(s) {
            return off;
        }
        let off = table.len() as u32;
        table.extend_from_slice(s.as_bytes());
        table.push(0);
        map.insert(s.to_string(), off);
        off
    }

    let mut records: Vec<u8> = Vec::with_capacity(models.len() * RECORD_LEN);
    for m in models {
        let name_off = intern(&m.name, &mut strings, &mut interned);
        let params_m = div_round(m.parameters_raw.unwrap_or(0), 1_000_000);
        let is_moe = m.is_moe.unwrap_or(false);
        let active_m = if is_moe {
            let a = div_round(m.active_parameters.unwrap_or(0), 1_000_000);
            // A sparse model with no stated active count still activates only
            // a slice. Fall back to the expert-count share rather than
            // charging it the full dense weight, which would be badly wrong
            // in the direction that matters (declaring it unrunnable).
            if a > 0 {
                a
            } else {
                let experts = m.num_experts.unwrap_or(8).max(2);
                (params_m / experts).max(params_m / 16).max(1)
            }
        } else {
            params_m
        };

        let has_arch = m.num_hidden_layers.is_some()
            && m.num_key_value_heads.is_some()
            && m.head_dim.is_some();

        let mut f: u16 = 0;
        if is_moe {
            f |= flags::MOE;
        }
        if has_arch {
            f |= flags::ARCH_METADATA;
        }
        if m.capabilities.iter().any(|c| c == "vision")
            || m.pipeline_tag.as_deref() == Some("image-text-to-text")
        {
            f |= flags::VISION;
        }
        if matches!(
            m.pipeline_tag.as_deref(),
            Some("feature-extraction") | Some("sentence-similarity") | Some("fill-mask")
        ) {
            f |= flags::EMBEDDING;
        }
        let lower = m.name.to_ascii_lowercase();
        if lower.contains("instruct") || lower.contains("-it") || lower.contains("chat") {
            f |= flags::INSTRUCT;
        }
        if m.gguf_sources
            .as_ref()
            .is_some_and(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
        {
            f |= flags::GGUF;
        }

        records.extend_from_slice(&name_off.to_le_bytes());
        records.extend_from_slice(&params_m.to_le_bytes());
        records.extend_from_slice(&active_m.to_le_bytes());
        records.extend_from_slice(&m.context_length.unwrap_or(4096).max(512).to_le_bytes());
        records.extend_from_slice(&clamp16(m.hidden_size.unwrap_or(0)).to_le_bytes());
        records.extend_from_slice(&clamp16(m.num_hidden_layers.unwrap_or(0)).to_le_bytes());
        records.extend_from_slice(&clamp16(m.num_key_value_heads.unwrap_or(0)).to_le_bytes());
        records.extend_from_slice(&clamp16(m.head_dim.unwrap_or(0)).to_le_bytes());
        records.extend_from_slice(&clamp16(m.vocab_size.unwrap_or(0) / 1000).to_le_bytes());
        records.extend_from_slice(&f.to_le_bytes());
        // Quantization mask: the scraper does not enumerate published GGUF
        // files per format, so zero is written meaning "assume the standard
        // ladder" rather than inventing availability the catalog cannot know.
        records.push(0);
        records.push(use_case_for(m) as u8);
        records.push(family_for(&lower, m.architecture.as_deref()));
        records.push(reputation_prior(m));
    }

    // Measurement section: 16 bytes each, already sorted by (hw_key, index)
    // so the reader can binary search without building anything.
    let mut meas = Vec::with_capacity(measurements.len() * MEASUREMENT_LEN);
    for m in measurements {
        meas.extend_from_slice(&m.hw_key.to_le_bytes());
        meas.extend_from_slice(&m.model_index.to_le_bytes());
        meas.extend_from_slice(&m.tps_x10.to_le_bytes());
        meas.extend_from_slice(&m.ttft_ms.to_le_bytes());
        meas.push(m.quant);
        meas.push(m.runs);
        meas.push(m.provider);
        meas.push(0); // flags, reserved
    }

    // Calibration section: 8 bytes each, sorted by hw_key.
    let mut cal = Vec::with_capacity(calibrations.len() * CALIBRATION_LEN);
    for (hw, factor, samples) in calibrations {
        cal.extend_from_slice(&hw.to_le_bytes());
        cal.extend_from_slice(&factor.to_le_bytes());
        cal.push(*samples);
        cal.push(0); // flags, reserved
    }

    let record_count = models.len() as u32;
    let rec_off = HEADER_LEN as u32;
    let str_off = rec_off + records.len() as u32;
    let meas_off = str_off + strings.len() as u32;
    let cal_off = meas_off + meas.len() as u32;

    let mut out =
        Vec::with_capacity(HEADER_LEN + records.len() + strings.len() + meas.len() + cal.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(RECORD_LEN as u16).to_le_bytes());
    out.extend_from_slice(&record_count.to_le_bytes());
    out.extend_from_slice(&rec_off.to_le_bytes());
    out.extend_from_slice(&str_off.to_le_bytes());
    out.extend_from_slice(&(strings.len() as u32).to_le_bytes());
    out.extend_from_slice(&meas_off.to_le_bytes());
    out.extend_from_slice(&(measurements.len() as u32).to_le_bytes());
    out.extend_from_slice(&cal_off.to_le_bytes());
    out.extend_from_slice(&(calibrations.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // flags, reserved
    out.extend_from_slice(&0u32.to_le_bytes()); // checksum, reserved
    debug_assert_eq!(out.len(), HEADER_LEN);
    out.extend_from_slice(&records);
    out.extend_from_slice(&strings);
    out.extend_from_slice(&meas);
    out.extend_from_slice(&cal);
    Ok(out)
}

fn use_case_for(m: &SourceModel) -> UseCase {
    if matches!(
        m.pipeline_tag.as_deref(),
        Some("feature-extraction") | Some("sentence-similarity") | Some("fill-mask")
    ) {
        return UseCase::Embedding;
    }
    if m.capabilities.iter().any(|c| c == "vision")
        || m.pipeline_tag.as_deref() == Some("image-text-to-text")
    {
        return UseCase::Multimodal;
    }
    let lower = m.name.to_ascii_lowercase();
    if lower.contains("coder") || lower.contains("code") || lower.contains("starcoder") {
        return UseCase::Coding;
    }
    if lower.contains("-r1")
        || lower.contains("reason")
        || lower.contains("think")
        || lower.contains("qwq")
    {
        return UseCase::Reasoning;
    }
    if lower.contains("chat") || lower.contains("instruct") {
        return UseCase::Chat;
    }
    UseCase::General
}

/// Family index, matched against the model name first and the architecture
/// string second. Names are the more reliable signal — `architecture` is often
/// the generic `llama` for anything using that block layout — so a name match
/// wins when both hit.
fn family_for(lower_name: &str, architecture: Option<&str>) -> u8 {
    for (i, fam) in FAMILIES.iter().enumerate().skip(1) {
        if lower_name.contains(fam) {
            return i as u8;
        }
    }
    if let Some(arch) = architecture {
        let arch = arch.to_ascii_lowercase();
        for (i, fam) in FAMILIES.iter().enumerate().skip(1) {
            if arch.contains(fam) {
                return i as u8;
            }
        }
    }
    0
}

/// A 0-255 prior derived from HuggingFace download and like counts.
///
/// This is popularity, not measured quality, and the name in the record is a
/// prior for exactly that reason: it nudges the ranking toward models people
/// actually run, and it is deliberately bounded to +/-15 points in the scorer
/// so it can never override parameter count or a bad fit.
fn reputation_prior(m: &SourceModel) -> u8 {
    let downloads = m.hf_downloads.unwrap_or(0) as f64;
    let likes = m.hf_likes.unwrap_or(0) as f64;
    // log10 of downloads saturating around 10^8, plus a smaller like term.
    let d = (downloads.max(1.0).log10() / 8.0).min(1.0);
    let l = (likes.max(1.0).log10() / 4.0).min(1.0);
    let combined = d * 0.75 + l * 0.25;
    (combined * 255.0).clamp(0.0, 255.0) as u8
}

fn div_round(value: u64, by: u64) -> u32 {
    (((value + by / 2) / by) as u32).max(if value > 0 { 1 } else { 0 })
}

fn clamp16(v: u32) -> u16 {
    v.min(u16::MAX as u32) as u16
}

fn inspect(args: &Args) -> Result<(), String> {
    let path = args
        .path
        .as_deref()
        .or(args.input.as_deref())
        .ok_or("inspect needs a path")?;
    let bytes = std::fs::read(path).map_err(|e| format!("reading {path}: {e}"))?;
    let catalog =
        Catalog::parse(&bytes).map_err(|e| format!("{path} is not a valid catalog: {e:?}"))?;

    println!("{path}: {} models, {} bytes", catalog.len(), bytes.len());
    println!(
        "  {} real measurements, {} machines with a calibration factor",
        catalog.measurement_count(),
        catalog.calibration_count()
    );
    let mut moe = 0usize;
    let mut with_arch = 0usize;
    for m in catalog.iter() {
        if m.is_moe() {
            moe += 1;
        }
        if m.has_arch_metadata() {
            with_arch += 1;
        }
    }
    println!(
        "  {moe} sparse, {with_arch} with full architecture metadata ({}%)",
        with_arch * 100 / catalog.len().max(1)
    );
    println!("\n  first 10 entries:");
    for m in catalog.iter().take(10) {
        println!(
            "    {:<44} {:>7}M params  {:>7} ctx  {}",
            m.name(),
            m.params_m(),
            m.context_length(),
            m.use_case().name()
        );
    }
    Ok(())
}
