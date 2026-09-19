//! Behavioural tests for the fit engine.
//!
//! These pin the parts that are policy rather than arithmetic — the verdict
//! bands, the run-mode caps, the order of the quantization walk — because
//! those are the things a well-meaning refactor silently changes.

use openchoice_core::catalog::{base_name, flags, FORMAT_VERSION, HEADER_LEN, MAGIC, RECORD_LEN};
use openchoice_core::{
    evaluate_model, recommend, Backend, Catalog, Filter, Hardware, KvQuant, Opts, Quant, RunMode,
    UseCase, Verdict,
};

/// Minimal in-test catalog builder, so the tests do not depend on the packer
/// binary or on a checked-in fixture that can drift from the format.
#[derive(Clone)]
struct Entry {
    name: &'static str,
    params_m: u32,
    active_m: u32,
    ctx: u32,
    layers: u16,
    kv_heads: u16,
    head_dim: u16,
    flags: u16,
    use_case: UseCase,
    quality: u8,
}

fn entry(name: &'static str, params_m: u32, ctx: u32) -> Entry {
    Entry {
        name,
        params_m,
        active_m: params_m,
        ctx,
        layers: 32,
        kv_heads: 8,
        head_dim: 128,
        flags: flags::ARCH_METADATA,
        use_case: UseCase::General,
        quality: 128,
    }
}

fn build(entries: &[Entry]) -> Vec<u8> {
    let mut strings = Vec::new();
    let mut records = Vec::new();
    for e in entries {
        let off = strings.len() as u32;
        strings.extend_from_slice(e.name.as_bytes());
        strings.push(0);

        records.extend_from_slice(&off.to_le_bytes());
        records.extend_from_slice(&e.params_m.to_le_bytes());
        records.extend_from_slice(&e.active_m.to_le_bytes());
        records.extend_from_slice(&e.ctx.to_le_bytes());
        records.extend_from_slice(&0u16.to_le_bytes()); // hidden
        records.extend_from_slice(&e.layers.to_le_bytes());
        records.extend_from_slice(&e.kv_heads.to_le_bytes());
        records.extend_from_slice(&e.head_dim.to_le_bytes());
        records.extend_from_slice(&32u16.to_le_bytes()); // vocab_k
        records.extend_from_slice(&e.flags.to_le_bytes());
        records.push(0); // quant mask: assume standard ladder
        records.push(e.use_case as u8);
        records.push(0); // family
        records.push(e.quality);
    }

    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(RECORD_LEN as u16).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    out.extend_from_slice(&(HEADER_LEN as u32).to_le_bytes());
    out.extend_from_slice(&((HEADER_LEN + records.len()) as u32).to_le_bytes());
    out.extend_from_slice(&(strings.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&records);
    out.extend_from_slice(&strings);
    out
}

fn gpu(vram_mb: u32, bandwidth: u16) -> Hardware {
    Hardware {
        ram_mb: 64 * 1024,
        vram_mb,
        unified: false,
        backend: Backend::Cuda,
        gpu_bandwidth_gbps: bandwidth,
        ram_bandwidth_gbps: 0,
        tflops_fp16_x10: 0,
        os_reserve_mb: 2048,
    }
}

// --- catalog format ---------------------------------------------------------

#[test]
fn catalog_round_trips_through_the_packed_format() {
    let bytes = build(&[
        entry("meta/Llama-3-8B", 8030, 8192),
        entry("tiny", 500, 2048),
    ]);
    let catalog = Catalog::parse(&bytes).expect("valid catalog");

    assert_eq!(catalog.len(), 2);
    let m = catalog.get(0).unwrap();
    assert_eq!(m.name(), "meta/Llama-3-8B");
    assert_eq!(m.params_m(), 8030);
    assert_eq!(m.context_length(), 8192);
    assert_eq!(catalog.get(1).unwrap().name(), "tiny");
}

#[test]
fn a_corrupt_header_is_rejected_rather_than_read() {
    let mut bytes = build(&[entry("x", 1000, 4096)]);
    bytes[0] = b'X';
    assert!(Catalog::parse(&bytes).is_err());

    let mut truncated = build(&[entry("x", 1000, 4096)]);
    truncated.truncate(HEADER_LEN + 4);
    assert!(
        Catalog::parse(&truncated).is_err(),
        "a record section running past the buffer must not parse"
    );
}

#[test]
fn find_prefers_an_exact_match_over_a_substring() {
    let bytes = build(&[
        entry("org/Qwen3-8B-Instruct-abliterated", 8000, 8192),
        entry("Qwen3-8B", 8000, 8192),
    ]);
    let catalog = Catalog::parse(&bytes).unwrap();
    assert_eq!(catalog.find("Qwen3-8B").unwrap().name(), "Qwen3-8B");
}

// --- verdict bands ----------------------------------------------------------

#[test]
fn the_verdict_follows_pool_utilisation_and_nothing_else() {
    let bytes = build(&[entry("m", 7000, 2048)]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let model = catalog.get(0).unwrap();
    let opts = Opts {
        max_quant: Quant::Q8_0,
        ..Default::default()
    };

    let roomy = evaluate_model(&model, &gpu(24 * 1024, 1008), &opts);
    assert_eq!(roomy.fit.verdict, Verdict::Perfect);
    assert_eq!(roomy.fit.run_mode, RunMode::Gpu);
    assert_eq!(roomy.fit.quant, Quant::Q8_0);
    assert!(roomy.fit.utilization_pctx10 <= 600);

    // Same model, a pool it half fills: Good rather than Perfect, and still
    // wholly on the card.
    let snug = evaluate_model(&model, &gpu(11 * 1024, 1008), &opts);
    assert_eq!(snug.fit.run_mode, RunMode::Gpu);
    assert_eq!(snug.fit.verdict, Verdict::Good);
    assert!((600..=850).contains(&snug.fit.utilization_pctx10));
}

/// When VRAM runs out, spilling to system RAM is tried *before* giving up
/// quality. A 7B at Q8_0 split across a small card and plenty of RAM is a
/// better answer than the same 7B squeezed to Q3 to stay resident, and the
/// engine has to prefer it in that order.
#[test]
fn a_full_card_spills_to_ram_before_it_degrades_quality() {
    let bytes = build(&[entry("m", 7000, 2048)]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let model = catalog.get(0).unwrap();

    let small_card_big_box = Hardware {
        ram_mb: 64 * 1024,
        ..gpu(6 * 1024, 1008)
    };
    let rec = evaluate_model(&model, &small_card_big_box, &Opts::default());

    assert_eq!(rec.fit.run_mode, RunMode::CpuGpu);
    assert_eq!(
        rec.fit.quant,
        Quant::Q8_0,
        "quality is given up last, not first"
    );
    assert_eq!(rec.fit.verdict, Verdict::Good, "a split fit caps at Good");
}

/// With only one pool to draw on, the quantization walk is the only lever
/// left, and it must step down exactly as far as it needs to.
#[test]
fn a_single_pool_steps_down_the_quantization_hierarchy() {
    let bytes = build(&[entry("m", 7000, 2048)]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let model = catalog.get(0).unwrap();

    // 8 GiB CPU-only: 6 GiB usable, against ~7.1 GiB of Q8_0 weights.
    let rec = evaluate_model(
        &model,
        &Hardware::cpu_only(8 * 1024, false),
        &Opts::default(),
    );

    assert_eq!(rec.fit.run_mode, RunMode::Cpu);
    assert!(
        rec.fit.quant > Quant::Q8_0,
        "expected a step down the hierarchy, got {}",
        rec.fit.quant.name()
    );
    assert!(rec.fit.verdict.runnable());
    assert!(
        rec.fit.memory.resident_mb() <= rec.fit.pool_mb,
        "whatever it picked has to actually fit"
    );
}

#[test]
fn a_cpu_fit_never_reaches_perfect() {
    let bytes = build(&[entry("m", 3000, 4096)]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let model = catalog.get(0).unwrap();

    let cpu = Hardware::cpu_only(64 * 1024, false);
    let rec = evaluate_model(&model, &cpu, &Opts::default());

    assert_eq!(rec.fit.run_mode, RunMode::Cpu);
    assert_eq!(
        rec.fit.verdict,
        Verdict::Good,
        "a roomy CPU fit is Good, never Perfect — Perfect means it is on the accelerator"
    );
}

#[test]
fn nothing_fits_reports_too_tight_rather_than_an_error() {
    let bytes = build(&[entry("huge", 405_000, 8192)]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let rec = evaluate_model(
        &catalog.get(0).unwrap(),
        &gpu(8 * 1024, 900),
        &Opts::default(),
    );
    assert_eq!(rec.fit.verdict, Verdict::TooTight);
    // Still describes a real configuration so the user can see how far off it is.
    assert!(rec.fit.memory.resident_mb() > 0);
}

// --- quantization policy ----------------------------------------------------

#[test]
fn f16_is_not_considered_unless_asked_for() {
    let bytes = build(&[entry("m", 3000, 4096)]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let model = catalog.get(0).unwrap();
    let roomy = gpu(48 * 1024, 1008);

    let default = evaluate_model(&model, &roomy, &Opts::default());
    assert_eq!(
        default.fit.quant,
        Quant::Q8_0,
        "F16 doubles memory for a 0.2% quality gain; it must not win by default"
    );

    let explicit = evaluate_model(
        &model,
        &roomy,
        &Opts {
            max_quant: Quant::F16,
            ..Default::default()
        },
    );
    assert_eq!(explicit.fit.quant, Quant::F16);
}

#[test]
fn a_smaller_kv_cache_can_rescue_a_long_context_model() {
    // 8B with a 128k window: the KV cache, not the weights, is what kills it.
    let mut e = entry("long", 8000, 131_072);
    e.layers = 32;
    let bytes = build(&[e]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let model = catalog.get(0).unwrap();
    let hw = gpu(24 * 1024, 1008);

    let f16 = evaluate_model(&model, &hw, &Opts::default());
    let q4 = evaluate_model(
        &model,
        &hw,
        &Opts {
            kv_quant: KvQuant::Q4,
            ..Default::default()
        },
    );

    assert!(
        q4.fit.memory.kv_cache_mb < f16.fit.memory.kv_cache_mb,
        "a 4-bit KV cache must be smaller than an f16 one"
    );
    assert!(q4.fit.usable_context >= f16.fit.usable_context);
}

// --- memory arithmetic ------------------------------------------------------

#[test]
fn weight_footprint_matches_the_effective_bits_per_weight() {
    // 7B at Q4_K_M: 7e9 * 4.83 / 8 = 4.23 GB = 4031 MiB.
    let mb = openchoice_core::fit::weights_mb(7000, Quant::Q4KM);
    assert!(
        (4000..=4060).contains(&mb),
        "7B Q4_K_M should be ~4031 MiB, got {mb}"
    );

    // Q8_0 is 8.5 bpw, not 8: the block scales are real bytes.
    let q8 = openchoice_core::fit::weights_mb(7000, Quant::Q8_0);
    assert!(
        (7050..=7150).contains(&q8),
        "7B Q8_0 should be ~7093 MiB, got {q8}"
    );
}

#[test]
fn kv_cache_uses_model_metadata_when_it_exists() {
    let bytes = build(&[entry("m", 8000, 8192)]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let rec = evaluate_model(
        &catalog.get(0).unwrap(),
        &gpu(24 * 1024, 1008),
        &Opts::default(),
    );
    assert_eq!(
        rec.fit.memory.kv_source,
        openchoice_core::KvSource::Metadata
    );

    // 2 * 32 layers * 8 kv heads * 128 dim * 8192 tokens * 2 bytes = 1024 MiB.
    assert_eq!(rec.fit.memory.kv_cache_mb, 1024);
}

#[test]
fn a_model_without_metadata_says_its_kv_figure_is_estimated() {
    let mut e = entry("bare", 8000, 8192);
    e.flags = 0;
    e.layers = 0;
    e.kv_heads = 0;
    e.head_dim = 0;
    let bytes = build(&[e]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let rec = evaluate_model(
        &catalog.get(0).unwrap(),
        &gpu(24 * 1024, 1008),
        &Opts::default(),
    );

    assert_eq!(
        rec.fit.memory.kv_source,
        openchoice_core::KvSource::Estimated
    );
    assert!(rec.fit.memory.kv_cache_mb > 0);
}

// --- speed ------------------------------------------------------------------

/// Decode throughput against published llama.cpp measurements.
///
/// A bandwidth roofline is an upper bound with a constant derate, so it runs
/// optimistic on small models where per-token overhead (sampling, kernel
/// launch, the sequential layer walk) is a larger share of the budget. The
/// tolerance below is wide and deliberately stated rather than tuned away:
/// these are order-of-magnitude-plus estimates, which is exactly what
/// `EstimateMethod` exists to communicate.
#[test]
fn decode_estimates_land_near_published_measurements() {
    // (label, params_m, vram_mb, bandwidth_gbps, measured tok/s)
    let cases = [
        (
            "Llama-3-8B Q4_K_M on RTX 4090",
            8030u32,
            24 * 1024u32,
            1008u16,
            130.0f32,
        ),
        (
            "Llama-2-13B Q4_K_M on RTX 3090",
            13_000,
            24 * 1024,
            936,
            50.0,
        ),
        (
            "Llama-2-70B Q4_K_M on M2 Ultra",
            70_000,
            128 * 1024,
            800,
            11.0,
        ),
    ];

    for (label, params, vram, bw, measured) in cases {
        let bytes = build(&[entry("m", params, 4096)]);
        let catalog = Catalog::parse(&bytes).unwrap();
        let rec = evaluate_model(
            &catalog.get(0).unwrap(),
            &gpu(vram, bw),
            &Opts {
                max_quant: Quant::Q4KM,
                ..Default::default()
            },
        );
        let est = rec.speed.decode_tps_x10 as f32 / 10.0;
        let ratio = est / measured;
        assert!(
            (0.6..=1.9).contains(&ratio),
            "{label}: estimated {est:.1} tok/s against a measured {measured:.1} (ratio {ratio:.2})"
        );
    }
}

#[test]
fn prefill_is_not_estimated_without_fp16_throughput() {
    let bytes = build(&[entry("m", 7000, 4096)]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let model = catalog.get(0).unwrap();

    let unknown = evaluate_model(&model, &gpu(24 * 1024, 1008), &Opts::default());
    assert_eq!(
        unknown.speed.prefill_tps_x10, None,
        "an unknown prefill must be None, never 0.0 — those mean different things"
    );
    assert_eq!(unknown.speed.ttft_ms, None);

    let mut hw = gpu(24 * 1024, 1008);
    hw.tflops_fp16_x10 = 1654;
    let known = evaluate_model(&model, &hw, &Opts::default());
    assert!(known.speed.prefill_tps_x10.is_some());
    assert!(known.speed.ttft_ms.is_some());
}

#[test]
fn an_unknown_memory_system_falls_back_and_says_so() {
    let bytes = build(&[entry("m", 7000, 4096)]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let rec = evaluate_model(
        &catalog.get(0).unwrap(),
        &gpu(24 * 1024, 0),
        &Opts::default(),
    );
    assert_eq!(
        rec.speed.method,
        openchoice_core::EstimateMethod::BackendConstant
    );
    assert!(rec.speed.decode_tps_x10 > 0);
}

// --- MoE --------------------------------------------------------------------

#[test]
fn a_sparse_model_is_charged_for_its_active_parameters_when_offloading() {
    let mut e = entry("moe", 46_700, 8192);
    e.active_m = 12_900;
    e.flags |= flags::MOE;
    let bytes = build(&[e]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let model = catalog.get(0).unwrap();

    // Too big for the card whole, but the active slice fits.
    let rec = evaluate_model(&model, &gpu(24 * 1024, 1008), &Opts::default());
    assert_eq!(rec.fit.run_mode, RunMode::MoeOffload);
    assert!(rec.fit.memory.offloaded_mb > 0);
    assert!(
        rec.fit.memory.weights_mb < openchoice_core::fit::weights_mb(46_700, rec.fit.quant),
        "MoE offload should account for the resident slice, not the whole model"
    );
}

// --- ranking ----------------------------------------------------------------

#[test]
fn repackages_of_the_same_weights_collapse_to_one_row() {
    assert_eq!(
        base_name("unsloth/gpt-oss-20b-unsloth-bnb-4bit"),
        "gpt-oss-20b"
    );
    assert_eq!(
        base_name("mlx-community/gpt-oss-20b-MXFP4-Q8"),
        "gpt-oss-20b"
    );
    assert_eq!(base_name("openai/gpt-oss-20b"), "gpt-oss-20b");
    assert_eq!(base_name("Qwen/Qwen2.5-7B-Instruct"), "Qwen2.5-7B-Instruct");

    let bytes = build(&[
        entry("openai/gpt-oss-20b", 20_000, 8192),
        entry("unsloth/gpt-oss-20b-BF16", 20_000, 8192),
        entry("mlx-community/gpt-oss-20b-MXFP4-Q8", 20_000, 8192),
        entry("Qwen/Qwen2.5-7B-Instruct", 7000, 8192),
    ]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let top = recommend::<8>(
        &catalog,
        &gpu(48 * 1024, 1008),
        &Opts::default(),
        &Filter::default(),
    );

    assert_eq!(
        top.len(),
        2,
        "three spellings of gpt-oss-20b plus one Qwen should rank as two entries"
    );
}

#[test]
fn unrunnable_models_never_outrank_runnable_ones() {
    let bytes = build(&[
        entry("small-but-runs", 3000, 4096),
        entry("enormous", 405_000, 4096),
    ]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let top = recommend::<8>(
        &catalog,
        &gpu(8 * 1024, 900),
        &Opts::default(),
        &Filter {
            // Allow everything through the filter so ordering is what is tested.
            min_verdict: Some(Verdict::TooTight),
            ..Default::default()
        },
    );
    assert_eq!(top.get(0).unwrap().model.name(), "small-but-runs");
}

#[test]
fn bigger_models_score_higher_on_capability_when_both_fit() {
    let bytes = build(&[entry("a-7b", 7000, 8192), entry("a-32b", 32_000, 8192)]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let hw = gpu(48 * 1024, 1008);

    let small = evaluate_model(&catalog.get(0).unwrap(), &hw, &Opts::default());
    let large = evaluate_model(&catalog.get(1).unwrap(), &hw, &Opts::default());

    assert!(
        large.scores.quality > small.scores.quality,
        "quality must still separate a 32B from a 7B: got {} vs {}",
        large.scores.quality,
        small.scores.quality
    );
    assert!(
        small.scores.speed > large.scores.speed,
        "and the smaller model must win on speed"
    );
}

#[test]
fn the_requested_use_case_reweights_rather_than_filters() {
    let mut coder = entry("org/Qwen2.5-Coder-7B", 7000, 32_768);
    coder.use_case = UseCase::Coding;
    let general = entry("org/Generalist-7B", 7000, 32_768);
    let bytes = build(&[general, coder]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let hw = gpu(24 * 1024, 1008);

    let coding_opts = Opts {
        use_case: Some(UseCase::Coding),
        ..Default::default()
    };
    let top = recommend::<8>(&catalog, &hw, &coding_opts, &Filter::default());

    assert_eq!(
        top.len(),
        2,
        "the generalist must still be listed, not filtered out"
    );
    assert_eq!(
        top.get(0).unwrap().model.name(),
        "org/Qwen2.5-Coder-7B",
        "but the coding model should lead when coding is asked for"
    );
}

// --- GPU table --------------------------------------------------------------

#[test]
fn gpu_lookup_prefers_the_most_specific_match() {
    let (desktop, _) = openchoice_core::lookup_gpu("NVIDIA GeForce RTX 4090").unwrap();
    let (laptop, _) = openchoice_core::lookup_gpu("NVIDIA GeForce RTX 4090 Laptop GPU").unwrap();
    assert_eq!(desktop, 1008);
    assert_eq!(
        laptop, 576,
        "the laptop part must not inherit the desktop's bandwidth"
    );

    assert!(openchoice_core::lookup_gpu("Some Unreleased Card").is_none());
}
