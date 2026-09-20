//! Best-effort hardware detection.
//!
//! Everything here is allowed to fail. A machine where GPU detection comes up
//! empty still gets a correct CPU-only answer, which is better than refusing
//! to answer at all. Anything that could not be determined is reported as
//! such rather than filled in with a plausible-looking default.

use openchoice_core::{Backend, Hardware};
use std::process::Command;

pub struct Detected {
    pub hardware: Hardware,
    pub cpu_name: String,
    pub cpu_cores: usize,
    pub gpu_name: Option<String>,
    /// Detection steps that failed, for the report. Silent failure here is how
    /// a wrong recommendation becomes hard to debug.
    pub notes: Vec<String>,
}

pub fn detect() -> Detected {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();

    let ram_mb = (sys.total_memory() / 1_048_576) as u32;
    let cpu_cores = sysinfo::System::physical_core_count().unwrap_or(0);
    let cpu_name = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .unwrap_or_else(|| "unknown".into());

    let mut notes = Vec::new();
    let arm = cfg!(target_arch = "aarch64");
    let apple_silicon = cfg!(target_os = "macos") && arm;

    let (gpu_name, vram_mb, backend, unified) = if apple_silicon {
        // Unified memory: the GPU draws from the same pool as everything else.
        // macOS caps how much may be wired, and the conventional ceiling is
        // ~75% of physical RAM, so that is what is offered to the engine.
        (
            Some(cpu_name.clone()),
            (ram_mb as f32 * 0.75) as u32,
            Backend::Metal,
            true,
        )
    } else if let Some((name, vram)) = detect_nvidia(&mut notes) {
        (Some(name), vram, Backend::Cuda, false)
    } else if let Some((name, vram)) = detect_amd(&mut notes) {
        (Some(name), vram, Backend::Rocm, false)
    } else {
        notes.push("no accelerator detected; scoring against system RAM".into());
        (
            None,
            0,
            if arm {
                Backend::CpuArm
            } else {
                Backend::CpuX86
            },
            false,
        )
    };

    // Fill bandwidth and fp16 throughput from the table when the GPU is one we
    // can name. A miss is not an error: the engine falls back to a backend
    // constant and labels the estimate accordingly.
    let (mut bw, mut tflops) = (0u16, 0u16);
    if let Some(name) = gpu_name.as_deref() {
        match openchoice_core::lookup_gpu(name) {
            Some((b, t)) => {
                bw = b;
                tflops = t;
            }
            None => notes.push(format!(
                "{name} is not in the bandwidth table; speed is a backend-constant estimate"
            )),
        }
    }

    let hardware = Hardware {
        ram_mb,
        vram_mb,
        unified,
        backend,
        gpu_bandwidth_gbps: bw,
        ram_bandwidth_gbps: 0,
        tflops_fp16_x10: tflops,
        os_reserve_mb: if ram_mb > 16_384 { 4096 } else { 2048 },
        // Identity for the measurement lookup. Community submissions record
        // the accelerator name, so that is what has to be hashed here; a
        // CPU-only box has no such identity and simply finds nothing.
        hw_key: gpu_name
            .as_deref()
            .map(openchoice_core::hw_key)
            .unwrap_or(0),
    };

    Detected {
        hardware,
        cpu_name,
        cpu_cores,
        gpu_name,
        notes,
    }
}

/// Total VRAM across all NVIDIA devices, via `nvidia-smi`.
fn detect_nvidia(notes: &mut Vec<String>) -> Option<(String, u32)> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        notes.push("nvidia-smi present but returned an error".into());
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut name = None;
    let mut total = 0u32;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let mut parts = line.split(',');
        let n = parts.next()?.trim().to_string();
        let mb: u32 = parts.next()?.trim().parse().ok()?;
        if name.is_none() {
            name = Some(n);
        }
        total += mb;
    }
    let name = name?;
    if total == 0 {
        return None;
    }
    Some((name, total))
}

/// AMD via `rocm-smi`. Output formats differ across versions, so this parses
/// loosely and gives up cleanly rather than guessing.
fn detect_amd(notes: &mut Vec<String>) -> Option<(String, u32)> {
    let out = Command::new("rocm-smi")
        .args(["--showproductname", "--showmeminfo", "vram", "--csv"])
        .output()
        .ok()?;
    if !out.status.success() {
        notes.push("rocm-smi present but returned an error".into());
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut name = String::from("AMD GPU");
    let mut total_bytes: u64 = 0;
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("card series") || lower.contains("product name") {
            if let Some(v) = line.split(',').nth(1) {
                let v = v.trim();
                if !v.is_empty() {
                    name = v.to_string();
                }
            }
        }
        if lower.contains("vram total memory") {
            for field in line.split(',').skip(1) {
                if let Ok(b) = field.trim().parse::<u64>() {
                    total_bytes += b;
                    break;
                }
            }
        }
    }
    if total_bytes == 0 {
        return None;
    }
    Some((name, (total_bytes / 1_048_576) as u32))
}
