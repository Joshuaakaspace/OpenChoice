# OpenChoice

**Which open LLM actually runs on that machine — answered in 57 KB, on a chip with no operating system.**

OpenChoice takes a description of some hardware and tells you which models will
run on it, at which quantization, how fast, and how much context you will
really get. The whole engine is `no_std`, dependency-free, and allocates
nothing, so the same arithmetic runs in a desktop CLI and on a $3
microcontroller holding the catalog in flash.

```
openchoice> gpu apple m2 pro
apple m2 pro: 200 GB/s, 13.5 TFLOPS fp16
openchoice> vram 24576
openchoice> ram 32768
openchoice> go
MODEL                        QUANT   TOK/S      FIT
openai/gpt-oss-20b            Q8_0   22.3^    Good
openai/gpt-oss-safeguard-     Q8_0   21.1+    Good
OpenGVLab/InternVL3_5-GPT     Q8_0   21.3+    Good
Qwen/Qwen3.6-35B-A3B-FP8      Q6_K   36.0^Marginal
deepseek-ai/DeepSeek-R1-D     Q8_0   11.4^    Good
* measured here  ^ rescaled  + calibrated
```

Those marks matter. `^` means somebody actually ran that model on an M2 Pro and
the figure was rescaled to the quantization shown; `+` means the formula was
corrected using other measurements from that same machine. Nothing here is
presented as an observation unless it is one.

That is the on-device console, and it needs no network, host, or cloud. The
transcript above was produced by `cargo run -p openchoice-embedded --example
repl` — the same `Console` the firmware drives, run on a host so the output
here is real rather than transcribed. The firmware compiles and its size is
measured below; it has **not** been run on physical hardware yet.

---

## Why this exists

The fit question has a good answer already: [llmfit](https://github.com/AlexsJones/llmfit),
whose estimation model OpenChoice builds on directly and credits in
[NOTICE](NOTICE). But llmfit ships its catalog as a 13.4 MB JSON blob parsed at
startup and pulls in `sysinfo`, `ureq`, `ratatui`, and `axum`. That is the
right set of choices for a laptop tool and a disqualifying one for anything
smaller.

The insight OpenChoice is built on: **the fit answer is arithmetic, and the
catalog is a table.** Neither needs an operating system.

| | llmfit | OpenChoice |
|---|---|---|
| Catalog storage | ~890 bytes/model (JSON) | **~67 bytes/model** (packed) |
| Startup | parses 13.4 MB of JSON | reads bytes in place, no parsing |
| Engine dependencies | ~40 crates | **zero** |
| Heap required | yes | **none** |
| Runs on a microcontroller | no | **yes** — 60 KB code, 440 B `.bss` |
| Real benchmark data | yes | **yes** — 358 measurements, 5.7 KB |

The shipped 258 KB catalog holds 3,923 models — everything left after dropping
sub-1M-parameter uploads, models with under 1,000 downloads, and non-text
pipelines. Packing all 15,049 source entries would come to roughly 1 MB. The
honest headline is the per-model figure, about 14x smaller than the JSON.

---

## Measured footprint

Real numbers from `llvm-size` on the ESP32-C3 firmware in this repo, built
with the 1,500-model catalog:

| Section | Bytes | What it is |
|---|---:|---|
| `.text` | 60,298 | the entire engine, console, and serial plumbing |
| `.rodata` | 118,688 | packed catalog, 309 measurements, GPU bandwidth table |
| `.data` | 1,476 | |
| `.bss` | **440** | static RAM — the engine allocates nothing |

~180 KB of flash total. It fits on every ESP32 variant with room to spare, and
the full 3,923-model catalog would still leave a 4 MB part 90% empty.

Carrying real benchmark data costs 1.6 KB of code and 5 KB of flash, and
`.bss` does not move at all: the measurements are read in place from flash by
binary search, like everything else.

---

## Install

```sh
git clone https://github.com/Joshuaakaspace/OpenChoice.git
cd OpenChoice
cargo build --release
```

You need a catalog. Build one from any llmfit-schema `hf_models.json`:

```sh
cargo run --release -p openchoice-catalog -- build \
  -i hf_models.json -o catalog/openchoice.ocb
```

To fold in real benchmark measurements, point it at a directory of community
submissions as well:

```sh
cargo run --release -p openchoice-catalog -- build \
  -i hf_models.json --community path/to/community -o catalog/openchoice.ocb
```

---

## Desktop use

```sh
# what does this machine actually have?
openchoice system

# rank the catalog for it
openchoice recommend --limit 10

# rank for a machine you do not own
openchoice recommend --gpu "RTX 4090" --vram 24576 --ram 65536

# reweight for the job rather than filtering the list
openchoice recommend --use-case coding

# one model, in full
openchoice fit qwen2.5-coder-7b
```

`openchoice fit` shows the whole derivation, because a tok/s number you cannot
check is not worth much:

```
Qwen/Qwen2.5-Coder-7B-Instruct
  7.6B params, 32k native context, coding

  verdict     Perfect on GPU
  quant       Q8_0
  memory      10086 MiB of 24576 MiB pool (41.0%)
              weights 7717 MiB + KV 1792 MiB (from model config) + overhead 577 MiB
  context     32k usable of 32k advertised

  decode      61.3 tok/s  (roofline at 1008 GB/s, 55% efficiency)
  prefill     3800 tok/s, 8621 ms to first token at 32k context

  scores      quality 79  speed 95  fit 92  context 100  →  90
```

---

## Running it on an ESP32

```sh
# develop against the device console without a board
cargo run -p openchoice-embedded --example repl -- catalog/openchoice-tiny.ocb

# build and flash the real thing
cd firmware/esp32c3
cargo build --release
espflash flash --monitor
```

Full guide: [docs/esp32.md](docs/esp32.md).

Targets the ESP32-C3/C6/H2 (RISC-V) on stock Rust. The Xtensa parts (ESP32,
S2, S3) need the `esp-rs` toolchain fork but nothing in `openchoice-core`
changes — see the guide.

---

## How it decides

### The verdict is one number

Pool utilization in, verdict out. Nothing else feeds it.

| `resident / pool` | Verdict |
|---|---|
| ≤ 60% | **Perfect** |
| ≤ 85% | **Good** |
| ≤ 98% | **Marginal** |
| > 98% | **Too Tight** |

Then capped by run mode: only a model sitting wholly on the accelerator can be
Perfect. CPU, CPU+GPU, and MoE-offload cap at Good — they are genuinely
runnable, just not fast. The band stops at 98% because a pool filled to the
last percent leaves nothing for allocator slack and does not load in practice.

### Quality is given up last

When VRAM runs out, the engine spills to system RAM *before* it degrades
quantization. A 7B at Q8_0 split across a small card and plenty of RAM beats
the same 7B crushed to Q3 to stay resident. Only when there is a single pool
and nothing left to spill into does the quantization walk step down.

F16 is excluded by default. It costs roughly double the memory of Q8_0 to buy
a 0.2% quality difference, so letting it win the walk systematically
recommends a small model at full precision over a far more capable one at Q8.
Pass `--max-quant f16` if you want it considered.

### Speed is two different problems

**Decode** is memory-bandwidth-bound — every resident weight is read once per
token — so it is estimated as bytes-moved over real bandwidth, derated by
0.55. **Prefill** is compute-bound, roughly `2 × active_params` FLOPs per
prompt token, so bandwidth says nothing about it. When fp16 throughput for the
hardware is unknown, prefill and TTFT report `null`, never `0.0`. Those mean
different things and conflating them is how a spec sheet becomes a lie.

Accuracy, stated plainly: the roofline runs optimistic on small models, where
sampling and kernel-launch overhead are a larger share of the budget. The
[test suite](openchoice-core/tests/engine.rs) pins estimates against published
llama.cpp measurements within a ±40% band rather than tuning the tolerance
away.

### Measurements beat formulas, and say which they are

The catalog can carry real benchmark results, keyed to the machine they were
run on. Every speed reports which of four things it is, and they are never
blended:

| Mark | Method | What it means |
|---|---|---|
| `*` | `measured` | Someone ran this model, at this quantization, on this hardware |
| `^` | `measured-adjusted` | Measured here at a *different* quantization, rescaled by the change in weight bytes |
| `+` | `calibrated` | The formula, scaled by how wrong it has been on this hardware for other models |
| ` ` | `roofline` | The formula, with nothing measured behind it |
| `~` | `backend-constant` | Not even bandwidth is known for this machine |

The `^` distinction is not pedantry. Most submitted benchmarks are Q4_K_M
pulls, and the default quantization walk lands on Q8_0 — nearly twice the
bytes per token. Reporting the raw Q4 figure against a Q8 row put a 39 tok/s
measurement next to a configuration the formula placed at 2 tok/s. Now it is
rescaled and labelled, and the raw observation is still shown underneath.

`+` is what makes a handful of submissions pay off across the whole catalog: if
the roofline overshot by 40% on every model measured on some card, it is
overshooting the rest of the catalog on that card by roughly 40% too.

A measured time-to-first-token is reported as belonging to *that benchmark
run*, never re-attributed to the context length on screen — the prompt length
was not recorded, so claiming otherwise would be inventing data.

### Context you get, not context advertised

A model with a 262k window on a card that can hold 8k of KV cache is reported
as 8k usable. That distinction is the whole difference between a spec sheet
and an answer.

### Duplicates collapse

`openai/gpt-oss-20b`, `unsloth/gpt-oss-20b-BF16`, and
`mlx-community/gpt-oss-20b-MXFP4-Q8` are one recommendation, not three. A
top-10 that is eight spellings of one model is not a top-10.

---

## Layout

```
openchoice-core/       no_std, zero-dep, zero-alloc fit engine
openchoice-catalog/    JSON -> .ocb packer
openchoice-cli/        desktop CLI with hardware detection
openchoice-embedded/   no_std device console: parse a line, render a report
firmware/esp32c3/      ESP32-C3 firmware (excluded from the workspace)
catalog/               packed .ocb catalogs
docs/                  format spec, ESP32 guide
```

`openchoice-core` has no dependencies and `#![forbid(unsafe_code)]`. That is
enforced, not aspirational — it is what lets the same code build for
`riscv32imc-unknown-none-elf` and x86-64 from one source.

---

## Docs

- [Catalog format](docs/catalog-format.md) — the `.ocb` layout, byte by byte
- [ESP32 guide](docs/esp32.md) — flashing, variants, sizing the catalog

## Where the measurements come from

The shipped catalog folds in the community benchmark submissions collected by
[llmfit](https://github.com/AlexsJones/llmfit) — 358 aggregated measurements
across 42 machines, with 23 of those machines having enough samples to derive
a correction factor. Matching a submitted model name (`qwen3:8b`,
`Qwen3-8B-Q4_K_M.gguf`, an MLX repo path) to a catalog entry happens in the
packer, where it can be counted and inspected:

```
358 measurements from 358 submissions — 493 of 810 results matched a catalog model
42 machines seen, 23 with enough samples for a calibration factor
```

493 of 810 is not a rounding error, and it is worth saying why the rest miss.
About 140 are one submitter's private Ollama serving configs
(`gemma4-12b-coder-fable5-latest-enfixed-agent-gpu100`), which name no
published model. The remainder are genuine gaps in the normalizer. Run the
packer with `--show-unmatched` to see them; improving that list is a
self-contained contribution.

## Credits

Built on the estimation model from
[llmfit](https://github.com/AlexsJones/llmfit) by Alex Jones. See
[NOTICE](NOTICE) for exactly what was carried over and what is new. If you
want a full-featured desktop tool — TUI, web dashboard, REST API, community
benchmarks — use llmfit; it does much more than this.

## License

MIT. See [LICENSE](LICENSE).
