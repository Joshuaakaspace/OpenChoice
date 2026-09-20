//! The device-side application: parse a line, render a report, allocate
//! nothing.
//!
//! This crate is what turns `openchoice-core` into something a
//! microcontroller can actually present to a human. It owns no I/O — the
//! firmware hands it a line of text and a buffer to write into — so it builds
//! and unit-tests on the host exactly as it runs on the chip.
//!
//! The whole session state is a `Console`, which is a `Hardware` plus a few
//! option bytes. There is no heap, no `String`, and no dynamic dispatch; the
//! largest thing on the stack is the output buffer the caller chose.

#![no_std]
#![forbid(unsafe_code)]

use core::fmt::Write;

use openchoice_core::{
    evaluate_model, recommend, Backend, Catalog, EstimateMethod, Filter, Hardware, KvQuant, Opts,
    Recommendation, UseCase, Verdict,
};

/// A fixed-capacity `core::fmt::Write` sink.
///
/// Writes past the end are dropped and the overflow is recorded rather than
/// panicking: on a device with no console of its own, a truncated report is a
/// far better failure than a reset loop.
pub struct ReportBuf<const N: usize> {
    buf: [u8; N],
    len: usize,
    overflowed: bool,
}

impl<const N: usize> Default for ReportBuf<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> ReportBuf<N> {
    pub const fn new() -> Self {
        ReportBuf {
            buf: [0; N],
            len: 0,
            overflowed: false,
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
        self.overflowed = false;
    }

    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    /// True when output was truncated. Callers should surface this rather than
    /// silently showing a half-written table.
    pub const fn overflowed(&self) -> bool {
        self.overflowed
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<const N: usize> Write for ReportBuf<N> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let space = N - self.len;
        if bytes.len() > space {
            // Copy what fits, but stop on a character boundary so `as_str`
            // stays valid UTF-8.
            let mut take = space;
            while take > 0 && !s.is_char_boundary(take) {
                take -= 1;
            }
            self.buf[self.len..self.len + take].copy_from_slice(&bytes[..take]);
            self.len += take;
            self.overflowed = true;
            return Ok(());
        }
        self.buf[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        self.len += bytes.len();
        Ok(())
    }
}

/// What a command did, so the firmware knows whether to reprint the prompt.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
    /// Handled; the buffer holds output to show.
    Ok,
    /// The line was not understood; the buffer holds a hint.
    Unknown,
    /// Empty line; nothing to show.
    Empty,
}

/// A device session: the machine being asked about, plus query options.
pub struct Console {
    pub hardware: Hardware,
    pub opts: Opts,
    pub filter: Filter,
    pub limit: usize,
}

impl Default for Console {
    fn default() -> Self {
        Self::new()
    }
}

impl Console {
    /// Starts describing a plausible mid-range laptop rather than an empty
    /// machine, so a user who types `go` first sees a real answer and can
    /// adjust from there.
    pub const fn new() -> Console {
        Console {
            hardware: Hardware {
                ram_mb: 16 * 1024,
                vram_mb: 0,
                unified: false,
                backend: Backend::CpuX86,
                gpu_bandwidth_gbps: 0,
                ram_bandwidth_gbps: 0,
                tflops_fp16_x10: 0,
                os_reserve_mb: 2048,
                hw_key: 0,
            },
            opts: Opts {
                context: None,
                kv_quant: KvQuant::F16,
                efficiency: 0.55,
                allow_moe_offload: true,
                max_quant: openchoice_core::Quant::Q8_0,
                use_case: None,
            },
            filter: Filter {
                use_case: None,
                min_verdict: None,
                min_tps: None,
                min_context: None,
                max_params_m: None,
            },
            limit: 10,
        }
    }

    /// Handle one line of input.
    ///
    /// Commands are deliberately terse — this is typed over a serial link,
    /// often on a phone. Anything unrecognised prints the help rather than an
    /// error code.
    pub fn handle<const N: usize>(
        &mut self,
        line: &str,
        catalog: &Catalog<'_>,
        out: &mut ReportBuf<N>,
    ) -> Outcome {
        out.clear();
        let line = line.trim();
        if line.is_empty() {
            return Outcome::Empty;
        }

        let (cmd, rest) = split_once(line, ' ');
        let rest = rest.trim();

        match cmd {
            "help" | "?" => {
                help(out);
                Outcome::Ok
            }
            "ram" => self.set_u32(rest, out, |hw, v| hw.ram_mb = v, "ram"),
            "vram" => self.set_u32(rest, out, |hw, v| hw.vram_mb = v, "vram"),
            "bw" => self.set_u32(
                rest,
                out,
                |hw, v| hw.gpu_bandwidth_gbps = v.min(u16::MAX as u32) as u16,
                "bw",
            ),
            "tflops" => self.set_u32(
                rest,
                out,
                |hw, v| hw.tflops_fp16_x10 = (v * 10).min(u16::MAX as u32) as u16,
                "tflops",
            ),
            "limit" => {
                match parse_u32(rest) {
                    Some(v) => {
                        self.limit = (v as usize).clamp(1, 32);
                        let _ = write!(out, "limit = {}", self.limit);
                    }
                    None => {
                        let _ = write!(out, "limit needs a number");
                    }
                }
                Outcome::Ok
            }
            "gpu" => {
                self.set_gpu(rest, out);
                Outcome::Ok
            }
            "cpu" => {
                self.hardware.vram_mb = 0;
                self.hardware.gpu_bandwidth_gbps = 0;
                self.hardware.tflops_fp16_x10 = 0;
                self.hardware.unified = false;
                self.hardware.backend = Backend::CpuX86;
                self.hardware.hw_key = 0;
                let _ = write!(out, "scoring against system RAM only");
                Outcome::Ok
            }
            "use" => {
                match UseCase::parse(rest) {
                    Some(uc) => {
                        // Reweights the ranking rather than filtering it: a
                        // strong generalist is still a valid answer to "what
                        // should I use for coding".
                        self.opts.use_case = Some(uc);
                        let _ = write!(out, "use case = {}", uc.name());
                    }
                    None if rest == "any" || rest.is_empty() => {
                        self.opts.use_case = None;
                        let _ = write!(out, "use case = any");
                    }
                    None => {
                        let _ = write!(
                            out,
                            "use <general|coding|reasoning|chat|multimodal|embedding|any>"
                        );
                    }
                }
                Outcome::Ok
            }
            "ctx" => {
                self.opts.context = parse_u32(rest);
                match self.opts.context {
                    Some(c) => {
                        let _ = write!(out, "context capped at {c}");
                    }
                    None => {
                        let _ = write!(out, "context = model native");
                    }
                }
                Outcome::Ok
            }
            "kv" => {
                self.opts.kv_quant = match rest {
                    "q8" => KvQuant::Q8,
                    "q4" => KvQuant::Q4,
                    _ => KvQuant::F16,
                };
                let _ = write!(out, "kv cache = {}", self.opts.kv_quant.name());
                Outcome::Ok
            }
            "min" => {
                self.filter.min_verdict = match rest {
                    "perfect" => Some(Verdict::Perfect),
                    "good" => Some(Verdict::Good),
                    "marginal" => Some(Verdict::Marginal),
                    _ => None,
                };
                let _ = write!(
                    out,
                    "minimum fit = {}",
                    self.filter
                        .min_verdict
                        .map(Verdict::name)
                        .unwrap_or("runnable")
                );
                Outcome::Ok
            }
            "hw" => {
                self.show_hardware(out);
                Outcome::Ok
            }
            "go" | "list" => {
                self.rank(catalog, out);
                Outcome::Ok
            }
            "fit" => {
                self.one(rest, catalog, out);
                Outcome::Ok
            }
            _ => {
                let _ = write!(out, "? {cmd} — type `help`");
                Outcome::Unknown
            }
        }
    }

    fn set_u32<const N: usize>(
        &mut self,
        rest: &str,
        out: &mut ReportBuf<N>,
        apply: fn(&mut Hardware, u32),
        label: &str,
    ) -> Outcome {
        match parse_u32(rest) {
            Some(v) => {
                apply(&mut self.hardware, v);
                // Declaring VRAM implies an accelerator; leaving the backend
                // on CPU would silently ignore the number just typed.
                if label == "vram" && v > 0 && self.hardware.backend.is_cpu() {
                    self.hardware.backend = Backend::Cuda;
                }
                let _ = write!(out, "{label} = {v}");
                Outcome::Ok
            }
            None => {
                let _ = write!(out, "{label} needs a number");
                Outcome::Unknown
            }
        }
    }

    fn set_gpu<const N: usize>(&mut self, name: &str, out: &mut ReportBuf<N>) {
        if name.is_empty() {
            let _ = write!(out, "gpu <name>, e.g. `gpu rtx 4090`");
            return;
        }
        // Naming the machine is also what unlocks any real measurements taken
        // on it, which is worth more than the bandwidth lookup.
        self.hardware.hw_key = openchoice_core::hw_key(name);
        match openchoice_core::lookup_gpu(name) {
            Some((bw, tf)) => {
                self.hardware.gpu_bandwidth_gbps = bw;
                self.hardware.tflops_fp16_x10 = tf;
                if self.hardware.backend.is_cpu() {
                    self.hardware.backend = Backend::Cuda;
                }
                let _ = write!(
                    out,
                    "{name}: {bw} GB/s, {:.1} TFLOPS fp16",
                    tf as f32 / 10.0
                );
                if self.hardware.vram_mb == 0 {
                    let _ = write!(out, "\nset vram too, e.g. `vram 24576`");
                }
            }
            None => {
                self.hardware.gpu_bandwidth_gbps = 0;
                self.hardware.tflops_fp16_x10 = 0;
                let _ = write!(
                    out,
                    "{name} is not in the table — set `bw <GB/s>` for a real estimate"
                );
            }
        }
    }

    fn show_hardware<const N: usize>(&self, out: &mut ReportBuf<N>) {
        let hw = &self.hardware;
        let _ = writeln!(
            out,
            "ram    {} MiB ({} usable)\nvram   {} MiB{}\nback   {}",
            hw.ram_mb,
            hw.usable_ram_mb(),
            hw.vram_mb,
            if hw.unified { " unified" } else { "" },
            hw.backend.name()
        );
        let (bw, src) = hw.resolve_gpu_bandwidth();
        if bw > 0.0 {
            let _ = writeln!(out, "bw     {bw:.0} GB/s ({})", src.name());
        } else {
            let _ = writeln!(out, "bw     unknown — estimates use backend constant");
        }
        if hw.tflops_fp16_x10 > 0 {
            let _ = writeln!(out, "fp16   {:.1} TFLOPS", hw.tflops_fp16_x10 as f32 / 10.0);
        } else {
            let _ = writeln!(out, "fp16   unknown — no prefill estimate");
        }
        let _ = write!(out, "kv     {}", self.opts.kv_quant.name());
    }

    /// Rank the whole catalog. This is the pass that has to stay allocation
    /// free: on a 320 KB-of-RAM device there is nowhere to put 15,000
    /// intermediate results, so only the best `limit` are ever retained.
    fn rank<const N: usize>(&self, catalog: &Catalog<'_>, out: &mut ReportBuf<N>) {
        let top = recommend::<32>(catalog, &self.hardware, &self.opts, &self.filter);
        if top.is_empty() {
            let _ = write!(
                out,
                "nothing in the catalog runs on this hardware.\n\
                 try `kv q8`, `ctx 4096`, or more ram."
            );
            return;
        }

        let _ = writeln!(
            out,
            "{:<26}{:>8}{:>8}{:>9}",
            "MODEL", "QUANT", "TOK/S", "FIT"
        );
        let mut any_measured = false;
        for rec in top.iter().take(self.limit) {
            let tps = rec.speed.decode_tps_x10 as f32 / 10.0;
            // The same provenance mark the desktop table carries. A device
            // with a two-inch window has less room to explain itself, not
            // less duty to separate a measured number from a guessed one.
            let mark = match rec.speed.method {
                EstimateMethod::Measured => '*',
                EstimateMethod::MeasuredAdjusted => '^',
                EstimateMethod::Calibrated => '+',
                EstimateMethod::Roofline => ' ',
                EstimateMethod::BackendConstant => '~',
            };
            any_measured |= mark != ' ';
            let _ = writeln!(
                out,
                "{:<26}{:>8}{:>7.1}{}{:>8}",
                trunc(rec.model.name(), 25),
                rec.fit.quant.name(),
                tps,
                mark,
                rec.fit.verdict.name()
            );
        }
        if any_measured {
            let _ = writeln!(out, "* measured here  ^ rescaled  + calibrated");
        }
        if out.overflowed() {
            let _ = write!(out, "\n(truncated — lower `limit`)");
        }
    }

    fn one<const N: usize>(&self, query: &str, catalog: &Catalog<'_>, out: &mut ReportBuf<N>) {
        if query.is_empty() {
            let _ = write!(out, "fit <model name>");
            return;
        }
        let Some(model) = catalog.find(query) else {
            let _ = write!(out, "no model matching \"{query}\"");
            return;
        };
        let rec = evaluate_model(catalog, &model, &self.hardware, &self.opts);
        render_detail(&rec, out);
    }
}

/// The single-model report. Every number that is an inference rather than a
/// lookup says so, because on a 2-inch screen there is no room for a footnote
/// and no excuse for implying more precision than exists.
pub fn render_detail<const N: usize>(rec: &Recommendation<'_>, out: &mut ReportBuf<N>) {
    let f = &rec.fit;
    let _ = writeln!(out, "{}", rec.model.name());
    let _ = writeln!(
        out,
        "{} params{}\n",
        fmt_params(rec.model.params_m()),
        if rec.model.is_moe() { " (MoE)" } else { "" }
    );
    let _ = writeln!(out, "{} on {}", f.verdict.name(), f.run_mode.name());
    let _ = writeln!(out, "quant  {}", f.quant.name());
    let _ = writeln!(
        out,
        "mem    {} / {} MiB  ({:.0}%)",
        f.memory.resident_mb(),
        f.pool_mb,
        f.utilization_pctx10 as f32 / 10.0
    );
    let _ = writeln!(
        out,
        "  w {} + kv {} + oh {}",
        f.memory.weights_mb, f.memory.kv_cache_mb, f.memory.overhead_mb
    );
    if f.memory.kv_source == openchoice_core::KvSource::Estimated {
        let _ = writeln!(out, "  (kv estimated, no arch metadata)");
    }
    let _ = writeln!(
        out,
        "ctx    {} usable / {} native",
        fmt_tokens(f.usable_context),
        fmt_tokens(rec.model.context_length())
    );
    let _ = writeln!(
        out,
        "speed  {:.1} tok/s  [{}]",
        rec.speed.decode_tps_x10 as f32 / 10.0,
        rec.speed.method.name()
    );
    match rec.speed.ttft_ms {
        Some(ms) => {
            let _ = writeln!(out, "ttft   {ms} ms");
        }
        None => {
            let _ = writeln!(out, "ttft   not estimated");
        }
    }
    let _ = write!(
        out,
        "score  q{} s{} f{} c{} = {}",
        rec.scores.quality,
        rec.scores.speed,
        rec.scores.fit,
        rec.scores.context,
        rec.scores.composite / 100
    );
}

fn help<const N: usize>(out: &mut ReportBuf<N>) {
    let _ = write!(
        out,
        "ram <MiB>      system memory\n\
         vram <MiB>     accelerator memory\n\
         gpu <name>     look up bandwidth, e.g. rtx 4090\n\
         bw <GB/s>      set bandwidth directly\n\
         tflops <n>     fp16 throughput, enables ttft\n\
         cpu            no accelerator\n\
         use <case>     coding|reasoning|chat|...|any\n\
         ctx <tokens>   cap context\n\
         kv <f16|q8|q4> kv cache precision\n\
         min <level>    minimum fit to list\n\
         limit <n>      rows to show\n\
         hw             show current machine\n\
         go             rank the catalog\n\
         fit <model>    score one model"
    );
}

// --- small helpers ----------------------------------------------------------

fn split_once(s: &str, sep: char) -> (&str, &str) {
    match s.find(sep) {
        Some(i) => (&s[..i], &s[i + sep.len_utf8()..]),
        None => (s, ""),
    }
}

fn parse_u32(s: &str) -> Option<u32> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut acc: u32 = 0;
    for b in s.bytes() {
        if !b.is_ascii_digit() {
            return None;
        }
        acc = acc.checked_mul(10)?.checked_add((b - b'0') as u32)?;
    }
    Some(acc)
}

fn trunc(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn fmt_tokens(t: u32) -> TokenDisplay {
    TokenDisplay(t)
}

/// Formats token counts without allocating a `String` for them.
pub struct TokenDisplay(u32);

impl core::fmt::Display for TokenDisplay {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.0 >= 1024 {
            write!(f, "{}k", self.0 / 1024)
        } else {
            write!(f, "{}", self.0)
        }
    }
}

fn fmt_params(m: u32) -> ParamDisplay {
    ParamDisplay(m)
}

pub struct ParamDisplay(u32);

impl core::fmt::Display for ParamDisplay {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.0 >= 1000 {
            write!(f, "{:.1}B", self.0 as f32 / 1000.0)
        } else {
            write!(f, "{}M", self.0)
        }
    }
}
