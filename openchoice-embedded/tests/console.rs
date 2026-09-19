//! Tests for the device console.
//!
//! The point of keeping the console free of hardware dependencies is that it
//! can be exercised here, on a host, instead of by flashing a board and
//! squinting at a serial monitor.

use openchoice_core::catalog::{flags, FORMAT_VERSION, HEADER_LEN, MAGIC, RECORD_LEN};
use openchoice_core::{Backend, Catalog, UseCase};
use openchoice_embedded::{Console, Outcome, ReportBuf};

fn build(entries: &[(&str, u32, u32)]) -> Vec<u8> {
    let mut strings = Vec::new();
    let mut records = Vec::new();
    for &(name, params_m, ctx) in entries {
        let off = strings.len() as u32;
        strings.extend_from_slice(name.as_bytes());
        strings.push(0);

        records.extend_from_slice(&off.to_le_bytes());
        records.extend_from_slice(&params_m.to_le_bytes());
        records.extend_from_slice(&params_m.to_le_bytes());
        records.extend_from_slice(&ctx.to_le_bytes());
        records.extend_from_slice(&0u16.to_le_bytes());
        records.extend_from_slice(&32u16.to_le_bytes());
        records.extend_from_slice(&8u16.to_le_bytes());
        records.extend_from_slice(&128u16.to_le_bytes());
        records.extend_from_slice(&32u16.to_le_bytes());
        records.extend_from_slice(&flags::ARCH_METADATA.to_le_bytes());
        records.push(0);
        records.push(UseCase::General as u8);
        records.push(0);
        records.push(128);
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

fn fixture() -> Vec<u8> {
    build(&[
        ("meta/Llama-3-8B-Instruct", 8030, 8192),
        ("Qwen/Qwen2.5-3B-Instruct", 3090, 32_768),
        ("meta/Llama-3-70B-Instruct", 70_600, 8192),
    ])
}

#[test]
fn a_gpu_name_populates_bandwidth_and_compute() {
    let bytes = fixture();
    let catalog = Catalog::parse(&bytes).unwrap();
    let mut console = Console::new();
    let mut out: ReportBuf<1024> = ReportBuf::new();

    assert_eq!(
        console.handle("gpu rtx 4090", &catalog, &mut out),
        Outcome::Ok
    );
    assert_eq!(console.hardware.gpu_bandwidth_gbps, 1008);
    assert_eq!(console.hardware.tflops_fp16_x10, 1654);
    assert_eq!(
        console.hardware.backend,
        Backend::Cuda,
        "naming a GPU must switch off the CPU backend, or the number is ignored"
    );
    assert!(out.as_str().contains("1008"));
}

#[test]
fn an_unknown_gpu_says_so_instead_of_inventing_a_number() {
    let bytes = fixture();
    let catalog = Catalog::parse(&bytes).unwrap();
    let mut console = Console::new();
    let mut out: ReportBuf<1024> = ReportBuf::new();

    console.handle("gpu Voodoo 3dfx", &catalog, &mut out);
    assert_eq!(console.hardware.gpu_bandwidth_gbps, 0);
    assert!(
        out.as_str().contains("not in the table"),
        "got: {}",
        out.as_str()
    );
}

#[test]
fn setting_vram_switches_off_the_cpu_only_assumption() {
    let bytes = fixture();
    let catalog = Catalog::parse(&bytes).unwrap();
    let mut console = Console::new();
    let mut out: ReportBuf<1024> = ReportBuf::new();

    assert!(console.hardware.backend.is_cpu());
    console.handle("vram 24576", &catalog, &mut out);
    assert_eq!(console.hardware.vram_mb, 24576);
    assert!(!console.hardware.backend.is_cpu());
}

#[test]
fn a_full_session_produces_a_ranking() {
    let bytes = fixture();
    let catalog = Catalog::parse(&bytes).unwrap();
    let mut console = Console::new();
    let mut out: ReportBuf<2048> = ReportBuf::new();

    for line in ["gpu rtx 4090", "vram 24576", "ram 65536"] {
        console.handle(line, &catalog, &mut out);
    }
    console.handle("go", &catalog, &mut out);

    let report = out.as_str();
    assert!(report.contains("MODEL"), "got: {report}");
    assert!(report.contains("Llama-3-8B"), "got: {report}");
    assert!(!out.overflowed());
}

#[test]
fn a_single_model_report_names_every_estimate_it_makes() {
    let bytes = fixture();
    let catalog = Catalog::parse(&bytes).unwrap();
    let mut console = Console::new();
    let mut out: ReportBuf<2048> = ReportBuf::new();

    console.handle("gpu rtx 4090", &catalog, &mut out);
    console.handle("vram 24576", &catalog, &mut out);
    console.handle("fit Llama-3-8B", &catalog, &mut out);

    let report = out.as_str();
    assert!(report.contains("quant"), "got: {report}");
    assert!(
        report.contains("roofline"),
        "the method must be stated: {report}"
    );
    assert!(report.contains("ttft"), "got: {report}");
}

#[test]
fn an_unmatched_model_is_reported_not_guessed_at() {
    let bytes = fixture();
    let catalog = Catalog::parse(&bytes).unwrap();
    let mut console = Console::new();
    let mut out: ReportBuf<1024> = ReportBuf::new();

    console.handle("fit definitely-not-a-model", &catalog, &mut out);
    assert!(out.as_str().contains("no model matching"));
}

#[test]
fn an_unknown_command_points_at_help_rather_than_failing_silently() {
    let bytes = fixture();
    let catalog = Catalog::parse(&bytes).unwrap();
    let mut console = Console::new();
    let mut out: ReportBuf<1024> = ReportBuf::new();

    assert_eq!(
        console.handle("frobnicate", &catalog, &mut out),
        Outcome::Unknown
    );
    assert!(out.as_str().contains("help"));
    assert_eq!(console.handle("   ", &catalog, &mut out), Outcome::Empty);
}

/// The buffer is the only memory bound on the device, so overflowing it must
/// truncate cleanly and say it did — never panic, never emit invalid UTF-8.
#[test]
fn an_overflowing_report_truncates_and_admits_it() {
    let bytes = fixture();
    let catalog = Catalog::parse(&bytes).unwrap();
    let mut console = Console::new();
    let mut tiny: ReportBuf<24> = ReportBuf::new();

    console.handle("help", &catalog, &mut tiny);
    assert!(tiny.overflowed());
    assert_eq!(tiny.len(), 24);
    // Still valid UTF-8, which `as_str` would otherwise silently blank.
    assert!(!tiny.as_str().is_empty());
}

#[test]
fn the_use_case_command_reweights_and_does_not_filter() {
    let bytes = fixture();
    let catalog = Catalog::parse(&bytes).unwrap();
    let mut console = Console::new();
    let mut out: ReportBuf<2048> = ReportBuf::new();

    console.handle("use coding", &catalog, &mut out);
    assert_eq!(console.opts.use_case, Some(UseCase::Coding));
    assert_eq!(
        console.filter.use_case, None,
        "asking for coding must not hide every model not tagged as one"
    );

    console.handle("use any", &catalog, &mut out);
    assert_eq!(console.opts.use_case, None);
}

#[test]
fn ranking_on_hardware_nothing_fits_explains_the_way_out() {
    let bytes = build(&[("meta/Llama-3-405B", 405_000, 8192)]);
    let catalog = Catalog::parse(&bytes).unwrap();
    let mut console = Console::new();
    let mut out: ReportBuf<1024> = ReportBuf::new();

    console.handle("ram 2048", &catalog, &mut out);
    console.handle("go", &catalog, &mut out);

    let report = out.as_str();
    assert!(
        report.contains("nothing in the catalog runs"),
        "got: {report}"
    );
    assert!(
        report.contains("kv q8"),
        "it should suggest a lever: {report}"
    );
}
