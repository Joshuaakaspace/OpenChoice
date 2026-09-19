//! OpenChoice desktop CLI.
//!
//! A thin shell over `openchoice-core`. All of the reasoning lives in the
//! engine so the answers here and the answers on a microcontroller are the
//! same answers; this binary only detects hardware, loads a catalog, and
//! formats output.

mod detect;

use std::process::ExitCode;

use openchoice_core::{
    evaluate_model, recommend, Backend, Catalog, Filter, Hardware, Opts, Quant, Recommendation,
    UseCase, Verdict,
};

const DEFAULT_CATALOG: &str = "catalog/openchoice.ocb";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("recommend");

    let result = match command {
        "system" => cmd_system(),
        "recommend" | "" => cmd_recommend(&args),
        "fit" => cmd_fit(&args),
        "help" | "--help" | "-h" => {
            usage();
            Ok(())
        }
        other => Err(format!("unknown command: {other}. Try `openchoice help`.")),
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
    println!(
        "openchoice — which open LLMs will actually run on this machine\n\
         \n\
         USAGE:\n\
         \x20 openchoice system                     show detected hardware\n\
         \x20 openchoice recommend [options]        rank the catalog for this machine\n\
         \x20 openchoice fit <model> [options]      score one model\n\
         \n\
         OPTIONS:\n\
         \x20 --catalog <path>    .ocb catalog (default: {DEFAULT_CATALOG})\n\
         \x20 --use-case <name>   general | coding | reasoning | chat | multimodal | embedding\n\
         \x20 --limit <n>         how many to show (default 15)\n\
         \x20 --context <tokens>  cap the context used for the memory estimate\n\
         \x20 --min-fit <level>   marginal | good | perfect\n\
         \x20 --kv <f16|q8|q4>    KV-cache precision (default f16)\n\
         \x20 --ram <MiB>         override detected RAM\n\
         \x20 --vram <MiB>        override detected VRAM\n\
         \x20 --gpu <name>        score against a named GPU instead of this one\n\
         \x20 --json              machine-readable output\n\
         \n\
         The hardware overrides are the interesting part: `--gpu \"RTX 4090\" --vram 24576`\n\
         answers what would run on a machine you do not have."
    );
}

struct Cli {
    catalog: String,
    use_case: Option<UseCase>,
    only: Option<UseCase>,
    limit: usize,
    context: Option<u32>,
    min_fit: Option<Verdict>,
    opts: Opts,
    json: bool,
    ram_override: Option<u32>,
    vram_override: Option<u32>,
    gpu_override: Option<String>,
    positional: Option<String>,
}

fn parse_cli(args: &[String]) -> Result<Cli, String> {
    let mut cli = Cli {
        catalog: std::env::var("OPENCHOICE_CATALOG").unwrap_or_else(|_| DEFAULT_CATALOG.into()),
        use_case: None,
        only: None,
        limit: 15,
        context: None,
        min_fit: None,
        opts: Opts::default(),
        json: false,
        ram_override: None,
        vram_override: None,
        gpu_override: None,
        positional: None,
    };

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--catalog" => cli.catalog = next(&mut it, "--catalog")?,
            "--use-case" => {
                let v = next(&mut it, "--use-case")?;
                cli.use_case =
                    Some(UseCase::parse(&v).ok_or_else(|| format!("unknown use case: {v}"))?);
            }
            "--only" => {
                let v = next(&mut it, "--only")?;
                cli.only =
                    Some(UseCase::parse(&v).ok_or_else(|| format!("unknown use case: {v}"))?);
            }
            "--limit" => cli.limit = num(&mut it, "--limit")? as usize,
            "--context" => cli.context = Some(num(&mut it, "--context")?),
            "--min-fit" => {
                let v = next(&mut it, "--min-fit")?;
                cli.min_fit = Some(match v.to_ascii_lowercase().as_str() {
                    "marginal" => Verdict::Marginal,
                    "good" => Verdict::Good,
                    "perfect" => Verdict::Perfect,
                    other => return Err(format!("unknown fit level: {other}")),
                });
            }
            "--kv" => {
                let v = next(&mut it, "--kv")?;
                cli.opts.kv_quant = match v.to_ascii_lowercase().as_str() {
                    "f16" | "fp16" => openchoice_core::KvQuant::F16,
                    "q8" | "q8_0" => openchoice_core::KvQuant::Q8,
                    "q4" | "q4_0" => openchoice_core::KvQuant::Q4,
                    other => return Err(format!("unknown KV precision: {other}")),
                };
            }
            "--max-quant" => {
                let v = next(&mut it, "--max-quant")?;
                cli.opts.max_quant =
                    Quant::parse(&v).ok_or_else(|| format!("unknown quantization: {v}"))?;
            }
            "--efficiency" => {
                let v = next(&mut it, "--efficiency")?;
                cli.opts.efficiency = v
                    .parse()
                    .map_err(|_| "--efficiency must be a number like 0.55".to_string())?;
            }
            "--ram" => cli.ram_override = Some(num(&mut it, "--ram")?),
            "--vram" => cli.vram_override = Some(num(&mut it, "--vram")?),
            "--gpu" => cli.gpu_override = Some(next(&mut it, "--gpu")?),
            "--json" => cli.json = true,
            other if !other.starts_with('-') && cli.positional.is_none() => {
                cli.positional = Some(other.to_string())
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    cli.opts.context = cli.context;
    cli.opts.use_case = cli.use_case;
    Ok(cli)
}

fn next<'a>(it: &mut impl Iterator<Item = &'a String>, flag: &str) -> Result<String, String> {
    it.next()
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn num<'a>(it: &mut impl Iterator<Item = &'a String>, flag: &str) -> Result<u32, String> {
    next(it, flag)?
        .parse()
        .map_err(|_| format!("{flag} must be a number"))
}

/// Apply `--ram`, `--vram`, and `--gpu` on top of what was detected.
///
/// Overriding the GPU also replaces bandwidth and fp16 throughput, otherwise
/// the estimate would pair one card's name with another's memory system and
/// quietly produce a number for a machine that does not exist.
fn apply_overrides(mut hw: Hardware, cli: &Cli) -> (Hardware, Option<String>) {
    if let Some(ram) = cli.ram_override {
        hw.ram_mb = ram;
    }
    let mut name = None;
    if let Some(gpu) = cli.gpu_override.as_deref() {
        name = Some(gpu.to_string());
        match openchoice_core::lookup_gpu(gpu) {
            Some((bw, tf)) => {
                hw.gpu_bandwidth_gbps = bw;
                hw.tflops_fp16_x10 = tf;
            }
            None => {
                hw.gpu_bandwidth_gbps = 0;
                hw.tflops_fp16_x10 = 0;
            }
        }
        let lower = gpu.to_ascii_lowercase();
        hw.unified = lower.starts_with('m') && lower.len() <= 8 || lower.contains("apple");
        hw.backend = if hw.unified {
            Backend::Metal
        } else if lower.contains("radeon") || lower.contains("rx ") || lower.contains("mi3") {
            Backend::Rocm
        } else if lower.contains("arc") {
            Backend::Sycl
        } else {
            Backend::Cuda
        };
    }
    if let Some(vram) = cli.vram_override {
        hw.vram_mb = vram;
        if hw.backend.is_cpu() {
            hw.backend = Backend::Cuda;
        }
    }
    (hw, name)
}

fn load_catalog(path: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| {
        format!(
            "cannot read catalog at {path}: {e}\n\
             Build one with:\n  \
             cargo run -p openchoice-catalog -- build -i models.json -o {path}"
        )
    })
}

fn cmd_system() -> Result<(), String> {
    let d = detect::detect();
    let hw = d.hardware;
    println!("CPU       {} ({} physical cores)", d.cpu_name, d.cpu_cores);
    println!(
        "RAM       {} MiB total, {} MiB usable after OS reserve",
        hw.ram_mb,
        hw.usable_ram_mb()
    );
    match d.gpu_name.as_deref() {
        Some(g) => println!(
            "GPU       {g} — {} MiB {}",
            hw.vram_mb,
            if hw.unified { "unified" } else { "dedicated" }
        ),
        None => println!("GPU       none detected"),
    }
    println!("Backend   {}", hw.backend.name());
    let (bw, src) = hw.resolve_gpu_bandwidth();
    if bw > 0.0 {
        println!("Bandwidth {bw:.0} GB/s ({})", src.name());
    } else {
        println!("Bandwidth unknown — speed estimates use the backend constant");
    }
    if hw.tflops_fp16_x10 > 0 {
        println!("fp16      {:.1} TFLOPS", hw.tflops_fp16_x10 as f32 / 10.0);
    } else {
        println!("fp16      unknown — prefill and TTFT will not be estimated");
    }
    for note in &d.notes {
        println!("note      {note}");
    }
    Ok(())
}

fn cmd_recommend(args: &[String]) -> Result<(), String> {
    let cli = parse_cli(args)?;
    let bytes = load_catalog(&cli.catalog)?;
    let catalog = Catalog::parse(&bytes).map_err(|e| format!("bad catalog: {e:?}"))?;

    let detected = detect::detect();
    let (hw, gpu_name) = apply_overrides(detected.hardware, &cli);
    let display_gpu = gpu_name.or(detected.gpu_name);

    let filter = Filter {
        use_case: cli.only,
        min_verdict: cli.min_fit,
        ..Default::default()
    };

    // 64 is the largest list anyone reads; keeping the bound fixed is what
    // lets the same call run on a device with no allocator.
    let top = recommend::<64>(&catalog, &hw, &cli.opts, &filter);

    if cli.json {
        print_json(top.iter().take(cli.limit));
        return Ok(());
    }

    println!(
        "{} — {} MiB RAM, {} MiB VRAM, {}",
        display_gpu.as_deref().unwrap_or("CPU only"),
        hw.ram_mb,
        hw.vram_mb,
        hw.backend.name()
    );
    if let Some(uc) = cli.use_case {
        println!("use case: {}", uc.name());
    }
    println!(
        "{} of {} models scored\n",
        top.len().min(cli.limit),
        catalog.len()
    );

    println!(
        "{:<40} {:>8} {:>9} {:>8} {:>10} {:>9} {:>6}",
        "MODEL", "QUANT", "MEMORY", "TOK/S", "FIT", "CONTEXT", "SCORE"
    );
    for rec in top.iter().take(cli.limit) {
        print_row(rec);
    }

    println!("\nlegend: MEMORY is resident footprint; CONTEXT is what actually fits, not the");
    println!("advertised window. Speed is a roofline estimate unless marked (~).");
    Ok(())
}

fn print_row(rec: &Recommendation) {
    let m = &rec.model;
    let tps = rec.speed.decode_tps_x10 as f32 / 10.0;
    let marker = match rec.speed.method {
        openchoice_core::EstimateMethod::Roofline => ' ',
        openchoice_core::EstimateMethod::BackendConstant => '~',
    };
    println!(
        "{:<40} {:>8} {:>7} MiB {:>7.1}{} {:>10} {:>9} {:>6}",
        truncate(m.name(), 40),
        rec.fit.quant.name(),
        rec.fit.memory.resident_mb(),
        tps,
        marker,
        rec.fit.verdict.name(),
        fmt_tokens(rec.fit.usable_context),
        rec.scores.composite / 100
    );
}

fn cmd_fit(args: &[String]) -> Result<(), String> {
    let cli = parse_cli(args)?;
    let query = cli
        .positional
        .as_deref()
        .ok_or("fit needs a model name, e.g. `openchoice fit qwen3-8b`")?;

    let bytes = load_catalog(&cli.catalog)?;
    let catalog = Catalog::parse(&bytes).map_err(|e| format!("bad catalog: {e:?}"))?;
    let model = catalog
        .find(query)
        .ok_or_else(|| format!("no model matching {query:?} in {}", cli.catalog))?;

    let detected = detect::detect();
    let (hw, _) = apply_overrides(detected.hardware, &cli);
    let rec = evaluate_model(&model, &hw, &cli.opts);

    if cli.json {
        print_json(std::iter::once(&rec));
        return Ok(());
    }

    let f = &rec.fit;
    println!("{}", model.name());
    println!(
        "  {} params{}, {} native context, {}",
        fmt_params(model.params_m()),
        if model.is_moe() {
            format!(" ({} active)", fmt_params(model.active_params_m()))
        } else {
            String::new()
        },
        fmt_tokens(model.context_length()),
        model.use_case().name()
    );
    println!();
    println!(
        "  verdict     {} on {}",
        f.verdict.name(),
        f.run_mode.name()
    );
    println!("  quant       {}", f.quant.name());
    println!(
        "  memory      {} MiB of {} MiB pool ({:.1}%)",
        f.memory.resident_mb(),
        f.pool_mb,
        f.utilization_pctx10 as f32 / 10.0
    );
    println!(
        "              weights {} MiB + KV {} MiB ({}) + overhead {} MiB",
        f.memory.weights_mb,
        f.memory.kv_cache_mb,
        match f.memory.kv_source {
            openchoice_core::KvSource::Metadata => "from model config",
            openchoice_core::KvSource::Estimated => "estimated from size class",
        },
        f.memory.overhead_mb
    );
    if f.memory.offloaded_mb > 0 {
        println!(
            "              {} MiB of inactive experts offloaded to RAM",
            f.memory.offloaded_mb
        );
    }
    println!(
        "  context     {} usable of {} advertised",
        fmt_tokens(f.usable_context),
        fmt_tokens(model.context_length())
    );
    println!();
    println!(
        "  decode      {:.1} tok/s  ({} at {} GB/s, {:.0}% efficiency)",
        rec.speed.decode_tps_x10 as f32 / 10.0,
        rec.speed.method.name(),
        rec.speed.bandwidth_gbps,
        rec.speed.efficiency * 100.0
    );
    match (rec.speed.prefill_tps_x10, rec.speed.ttft_ms) {
        (Some(p), Some(t)) => println!(
            "  prefill     {:.0} tok/s, {} ms to first token at {} context",
            p as f32 / 10.0,
            t,
            fmt_tokens(f.effective_context)
        ),
        _ => println!("  prefill     not estimated (fp16 throughput unknown for this hardware)"),
    }
    println!();
    println!(
        "  scores      quality {}  speed {}  fit {}  context {}  →  {}",
        rec.scores.quality,
        rec.scores.speed,
        rec.scores.fit,
        rec.scores.context,
        rec.scores.composite / 100
    );
    Ok(())
}

fn print_json<'a>(recs: impl Iterator<Item = &'a Recommendation<'a>>) {
    println!("{{\"models\":[");
    let mut first = true;
    for rec in recs {
        if !first {
            println!(",");
        }
        first = false;
        let f = &rec.fit;
        print!(
            "  {{\"name\":\"{}\",\"params_m\":{},\"quant\":\"{}\",\"verdict\":\"{}\",\
             \"run_mode\":\"{}\",\"memory_mb\":{},\"pool_mb\":{},\"utilization_pct\":{:.1},\
             \"decode_tps\":{:.1},\"estimate_method\":\"{}\",\"bandwidth_gbps\":{},\
             \"usable_context\":{},\"native_context\":{},\"kv_source\":\"{}\",\
             \"prefill_tps\":{},\"ttft_ms\":{},\
             \"scores\":{{\"quality\":{},\"speed\":{},\"fit\":{},\"context\":{},\"composite\":{}}}}}",
            escape(rec.model.name()),
            rec.model.params_m(),
            f.quant.name(),
            f.verdict.name(),
            f.run_mode.name(),
            f.memory.resident_mb(),
            f.pool_mb,
            f.utilization_pctx10 as f32 / 10.0,
            rec.speed.decode_tps_x10 as f32 / 10.0,
            rec.speed.method.name(),
            rec.speed.bandwidth_gbps,
            f.usable_context,
            rec.model.context_length(),
            match f.memory.kv_source {
                openchoice_core::KvSource::Metadata => "metadata",
                openchoice_core::KvSource::Estimated => "estimated",
            },
            // null, not 0: "not estimated" is not "instant".
            rec.speed
                .prefill_tps_x10
                .map(|v| format!("{:.1}", v as f32 / 10.0))
                .unwrap_or_else(|| "null".into()),
            rec.speed
                .ttft_ms
                .map(|v| v.to_string())
                .unwrap_or_else(|| "null".into()),
            rec.scores.quality,
            rec.scores.speed,
            rec.scores.fit,
            rec.scores.context,
            rec.scores.composite
        );
    }
    println!("\n]}}");
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        // Do not split a multi-byte character.
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

fn fmt_tokens(t: u32) -> String {
    if t >= 1000 {
        format!("{}k", t / 1024)
    } else {
        t.to_string()
    }
}

fn fmt_params(m: u32) -> String {
    if m >= 1000 {
        format!("{:.1}B", m as f32 / 1000.0)
    } else {
        format!("{m}M")
    }
}
