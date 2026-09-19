//! The hardware description the engine scores against.
//!
//! This is a plain value type with no detection in it. The CLI fills it from
//! `sysinfo` and vendor tools; the ESP32 firmware fills it from whatever the
//! user typed into the serial console or the captive portal. The engine cannot
//! tell the difference, which is the point: the same arithmetic answers "what
//! runs on this machine" and "what would run on that machine over there".

/// Acceleration backend. Determines the fallback throughput constant when no
/// memory bandwidth is known.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Backend {
    Cuda = 0,
    Metal = 1,
    Rocm = 2,
    Sycl = 3,
    CpuArm = 4,
    CpuX86 = 5,
    Npu = 6,
    Vulkan = 7,
}

impl Backend {
    pub const fn name(self) -> &'static str {
        match self {
            Backend::Cuda => "CUDA",
            Backend::Metal => "Metal",
            Backend::Rocm => "ROCm",
            Backend::Sycl => "SYCL",
            Backend::CpuArm => "CPU (ARM)",
            Backend::CpuX86 => "CPU (x86)",
            Backend::Npu => "NPU",
            Backend::Vulkan => "Vulkan",
        }
    }

    pub const fn is_cpu(self) -> bool {
        matches!(self, Backend::CpuArm | Backend::CpuX86)
    }

    /// Throughput constant K for the no-bandwidth fallback `K / params_b`.
    /// Only used when neither an explicit bandwidth nor a GPU-table hit is
    /// available; a roofline estimate from real bandwidth is always better.
    pub const fn fallback_k(self) -> f32 {
        match self {
            Backend::Cuda => 220.0,
            Backend::Metal => 160.0,
            Backend::Rocm => 180.0,
            Backend::Sycl => 100.0,
            Backend::Vulkan => 110.0,
            Backend::CpuArm => 90.0,
            Backend::CpuX86 => 70.0,
            Backend::Npu => 390.0,
        }
    }

    pub const fn from_u8(v: u8) -> Option<Backend> {
        match v {
            0 => Some(Backend::Cuda),
            1 => Some(Backend::Metal),
            2 => Some(Backend::Rocm),
            3 => Some(Backend::Sycl),
            4 => Some(Backend::CpuArm),
            5 => Some(Backend::CpuX86),
            6 => Some(Backend::Npu),
            7 => Some(Backend::Vulkan),
            _ => None,
        }
    }
}

/// Where a bandwidth figure came from. Reported alongside every estimate so a
/// number can be traced to its source rather than taken on faith.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BandwidthSource {
    /// Caller supplied it explicitly.
    Override,
    /// Matched a name in the built-in GPU table.
    GpuTable,
    /// No bandwidth known; the backend constant was used instead.
    BackendConstant,
}

impl BandwidthSource {
    pub const fn name(self) -> &'static str {
        match self {
            BandwidthSource::Override => "override",
            BandwidthSource::GpuTable => "gpu-table",
            BandwidthSource::BackendConstant => "backend-constant",
        }
    }
}

/// A machine to score models against.
///
/// Memory is in MiB throughout. Bandwidth is in GB/s using the vendor
/// convention (10^9 bytes/s), because that is how every spec sheet the table
/// was built from reports it.
#[derive(Clone, Copy, Debug)]
pub struct Hardware {
    /// Total system RAM, MiB.
    pub ram_mb: u32,
    /// Total VRAM across all accelerators, MiB. Zero means CPU-only.
    pub vram_mb: u32,
    /// True when VRAM and RAM are the same physical pool (Apple Silicon,
    /// AMD APUs, Jetson). Prevents double-counting the pool in CPU+GPU mode.
    pub unified: bool,
    pub backend: Backend,
    /// Accelerator memory bandwidth, GB/s. Zero means unknown.
    pub gpu_bandwidth_gbps: u16,
    /// System memory bandwidth, GB/s. Zero falls back to a DDR default.
    pub ram_bandwidth_gbps: u16,
    /// fp16 dense matmul throughput, TFLOPS x10. Zero means unknown, which
    /// makes prefill and TTFT report `None` rather than a fabricated zero.
    pub tflops_fp16_x10: u16,
    /// Memory the OS and other processes are assumed to hold, MiB. Subtracted
    /// from the CPU pool before any verdict is computed.
    pub os_reserve_mb: u32,
}

impl Hardware {
    /// A CPU-only machine with the given RAM. The honest default for anything
    /// we know nothing else about.
    pub const fn cpu_only(ram_mb: u32, arm: bool) -> Hardware {
        Hardware {
            ram_mb,
            vram_mb: 0,
            unified: false,
            backend: if arm {
                Backend::CpuArm
            } else {
                Backend::CpuX86
            },
            gpu_bandwidth_gbps: 0,
            ram_bandwidth_gbps: 0,
            tflops_fp16_x10: 0,
            os_reserve_mb: 2048,
        }
    }

    /// RAM usable for inference after the OS reserve.
    pub const fn usable_ram_mb(&self) -> u32 {
        self.ram_mb.saturating_sub(self.os_reserve_mb)
    }

    /// Effective system-memory bandwidth, GB/s. DDR4-3200 dual channel is the
    /// default because it is the most common thing sitting under a discrete
    /// GPU, and it is the number that governs CPU-offload speed.
    pub fn effective_ram_bandwidth(&self) -> f32 {
        if self.ram_bandwidth_gbps > 0 {
            self.ram_bandwidth_gbps as f32
        } else if self.unified {
            // A unified system with no stated bandwidth is still far better
            // than socketed DDR; Apple's slowest unified parts start here.
            100.0
        } else {
            51.2
        }
    }

    /// Resolve accelerator bandwidth and say where the number came from.
    ///
    /// A zero bandwidth with `BackendConstant` means "nothing known" and tells
    /// the speed model to use the `K / params_b` fallback instead of a
    /// roofline it cannot actually compute.
    pub fn resolve_gpu_bandwidth(&self) -> (f32, BandwidthSource) {
        if self.gpu_bandwidth_gbps > 0 {
            (self.gpu_bandwidth_gbps as f32, BandwidthSource::Override)
        } else if self.unified {
            (
                self.effective_ram_bandwidth(),
                BandwidthSource::BackendConstant,
            )
        } else {
            (0.0, BandwidthSource::BackendConstant)
        }
    }
}

/// Memory bandwidth (GB/s) and fp16 dense TFLOPS for accelerators we can name.
///
/// Matching is substring-based on a lowercased name, longest pattern first, so
/// "NVIDIA GeForce RTX 4090" and "RTX 4090 Laptop GPU" do not collide. TFLOPS
/// are dense fp16 with fp32 accumulate, excluding sparsity claims, because
/// sparsity does not apply to a prefill matmul.
#[cfg(feature = "gpu-db")]
pub const GPU_TABLE: &[(&str, u16, u16)] = &[
    // pattern, bandwidth GB/s, fp16 TFLOPS x10
    // --- NVIDIA datacenter ---
    ("h200", 4800, 9890),
    ("h100 nvl", 3900, 9890),
    ("h100", 3350, 9890),
    ("a100 80gb", 2039, 3120),
    ("a100", 1555, 3120),
    ("l40s", 864, 3625),
    ("l40", 864, 1814),
    ("l4", 300, 1210),
    ("a40", 696, 1495),
    ("a30", 933, 1650),
    ("a10g", 600, 700),
    ("a10", 600, 1250),
    ("v100", 900, 1120),
    ("t4", 320, 650),
    // --- NVIDIA GeForce 50 series ---
    ("rtx 5090", 1792, 3186),
    ("rtx 5080", 960, 1710),
    ("rtx 5070 ti", 896, 1430),
    ("rtx 5070", 672, 1130),
    ("rtx 5060 ti", 448, 940),
    ("rtx 5060", 448, 780),
    // --- NVIDIA GeForce 40 series ---
    ("rtx 4090 laptop", 576, 1650),
    ("rtx 4090", 1008, 1654),
    ("rtx 4080 super", 736, 1300),
    ("rtx 4080 laptop", 432, 1000),
    ("rtx 4080", 717, 1230),
    ("rtx 4070 ti super", 672, 1000),
    ("rtx 4070 ti", 504, 800),
    ("rtx 4070 super", 504, 710),
    ("rtx 4070 laptop", 256, 500),
    ("rtx 4070", 504, 590),
    ("rtx 4060 ti", 288, 440),
    ("rtx 4060", 272, 380),
    ("rtx 4050", 192, 270),
    // --- NVIDIA GeForce 30 series ---
    ("rtx 3090 ti", 1008, 800),
    ("rtx 3090", 936, 710),
    ("rtx 3080 ti", 912, 680),
    ("rtx 3080", 760, 595),
    ("rtx 3070 ti", 608, 436),
    ("rtx 3070", 448, 407),
    ("rtx 3060 ti", 448, 324),
    ("rtx 3060", 360, 257),
    ("rtx 3050", 224, 180),
    // --- NVIDIA GeForce 20 / 16 ---
    ("rtx 2080 ti", 616, 268),
    ("rtx 2070", 448, 179),
    ("rtx 2060", 336, 129),
    ("gtx 1660", 192, 50),
    ("gtx 1080 ti", 484, 45),
    // --- NVIDIA professional / embedded ---
    ("rtx 6000 ada", 960, 1457),
    ("rtx a6000", 768, 387),
    ("rtx a5000", 768, 273),
    ("rtx a4000", 448, 192),
    ("jetson agx orin", 204, 1050),
    ("jetson orin nx", 102, 700),
    ("jetson orin nano", 68, 340),
    // --- Apple Silicon ---
    ("m4 max", 546, 340),
    ("m4 pro", 273, 170),
    ("m4", 120, 90),
    ("m3 ultra", 819, 570),
    ("m3 max", 400, 285),
    ("m3 pro", 150, 140),
    ("m3", 100, 70),
    ("m2 ultra", 800, 540),
    ("m2 max", 400, 270),
    ("m2 pro", 200, 135),
    ("m2", 100, 68),
    ("m1 ultra", 800, 420),
    ("m1 max", 400, 210),
    ("m1 pro", 200, 105),
    ("m1", 68, 52),
    // --- AMD ---
    ("mi300x", 5300, 13070),
    ("mi250x", 3276, 3830),
    ("mi210", 1638, 1810),
    ("rx 7900 xtx", 960, 1230),
    ("rx 7900 xt", 800, 1030),
    ("rx 7900 gre", 576, 920),
    ("rx 7800 xt", 624, 750),
    ("rx 7700 xt", 432, 630),
    ("rx 7600", 288, 430),
    ("rx 6900 xt", 512, 460),
    ("rx 6800 xt", 512, 414),
    ("rx 6800", 512, 366),
    ("rx 6700 xt", 384, 269),
    ("rx 6600", 224, 177),
    ("rx 9070 xt", 645, 1950),
    ("rx 9070", 645, 1700),
    ("rx 9060 xt", 320, 820),
    ("radeon 890m", 128, 300),
    ("radeon 780m", 102, 170),
    ("ryzen ai max", 256, 500),
    // --- Intel ---
    ("arc b580", 456, 456),
    ("arc b570", 380, 400),
    ("arc a770", 560, 393),
    ("arc a750", 512, 344),
    ("arc a380", 186, 100),
];

/// Look up bandwidth (GB/s) and fp16 TFLOPS x10 for a GPU name.
///
/// Returns the longest matching pattern so more specific entries win over
/// their own prefixes. An unrecognised name is not an error: the caller falls
/// back to the backend constant and reports that it did.
#[cfg(feature = "gpu-db")]
pub fn lookup_gpu(name: &str) -> Option<(u16, u16)> {
    let mut lower = [0u8; 96];
    let bytes = name.as_bytes();
    let n = if bytes.len() < 96 { bytes.len() } else { 96 };
    for i in 0..n {
        lower[i] = bytes[i].to_ascii_lowercase();
    }
    let hay = core::str::from_utf8(&lower[..n]).ok()?;

    let mut best: Option<(usize, u16, u16)> = None;
    for &(pat, bw, tf) in GPU_TABLE {
        if contains(hay, pat) {
            let better = match best {
                Some((len, _, _)) => pat.len() > len,
                None => true,
            };
            if better {
                best = Some((pat.len(), bw, tf));
            }
        }
    }
    best.map(|(_, bw, tf)| (bw, tf))
}

#[cfg(feature = "gpu-db")]
fn contains(hay: &str, needle: &str) -> bool {
    let h = hay.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || n.len() > h.len() {
        return false;
    }
    for start in 0..=(h.len() - n.len()) {
        if &h[start..start + n.len()] == n {
            return true;
        }
    }
    false
}
