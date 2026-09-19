# Running OpenChoice on an ESP32

The device holds the whole model catalog in flash and answers fit questions
over USB serial. No network, no host, no cloud.

## What you need

- An ESP32-C3, C6, or H2 board (RISC-V — these work on stock Rust)
- `espflash`: `cargo install espflash`
- The `riscv32imc-unknown-none-elf` target: `rustup target add riscv32imc-unknown-none-elf`

## Build and flash

```sh
cd firmware/esp32c3
cargo build --release
espflash flash --monitor
```

`.cargo/config.toml` already sets the target, the linker script, and
`espflash` as the runner, so `cargo run --release` flashes and opens a monitor
in one step.

## Using it

```
OpenChoice — 1500 models in 93 KB of flash
type `help`, or `go` to rank for the default machine

openchoice> gpu rtx 4090
rtx 4090: 1008 GB/s, 165.4 TFLOPS fp16
set vram too, e.g. `vram 24576`
openchoice> vram 24576
vram = 24576
openchoice> ram 65536
ram = 65536
openchoice> go
MODEL                          QUANT  TOK/S      FIT
openai/gpt-oss-20b              Q8_0   68.2     Good
microsoft/phi-4                 Q8_0   32.1     Good
...
```

| Command | Does |
|---|---|
| `ram <MiB>` / `vram <MiB>` | describe the machine being asked about |
| `gpu <name>` | look up bandwidth and fp16 throughput, e.g. `gpu rtx 4090` |
| `bw <GB/s>` / `tflops <n>` | set them directly for a card not in the table |
| `cpu` | no accelerator |
| `use <case>` | reweight for coding, reasoning, chat, … or `any` |
| `ctx <tokens>` / `kv <f16\|q8\|q4>` | cap context, shrink the KV cache |
| `min <level>` / `limit <n>` | filter and trim the listing |
| `hw` | show the machine as currently described |
| `go` | rank the catalog |
| `fit <model>` | full report for one model |

## Develop without a board

The console has no hardware dependency, so it runs on a host:

```sh
cargo run -p openchoice-embedded --example repl -- catalog/openchoice-tiny.ocb
```

Same `Console`, same `ReportBuf<2048>`, same output — including where it
truncates. This is the fast loop; flashing is for the demo.

## Footprint

Measured with `llvm-size` on the release build with the 1,500-model catalog:

| Section | Bytes |
|---|---:|
| `.text` | 58,706 |
| `.rodata` | 113,392 |
| `.data` | 1,420 |
| `.bss` | 440 |

**440 bytes of static RAM.** The engine never allocates: the catalog is read
in place out of flash, and the only sizeable buffer is the `ReportBuf<2048>`
the firmware owns on the stack. Ranking all 1,500 models keeps only the best
32 results, in a fixed array, in one pass.

### Sizing the catalog

The firmware embeds `catalog/openchoice-tiny.ocb` (1,500 models, 93 KB). To
change it, build a different one and rebuild:

```sh
cargo run -p openchoice-catalog -- build -i hf_models.json -o catalog/openchoice-tiny.ocb --top 3000
```

Even the full 252 KB catalog leaves a 4 MB part around 90% empty, so `--top`
is about taste, not necessity, unless your partition table is unusual.

## Xtensa parts (ESP32, ESP32-S2, ESP32-S3)

`openchoice-core` and `openchoice-embedded` are plain `no_std` Rust with zero
dependencies and no target-specific code, so they build for Xtensa unchanged.
What Xtensa needs is a toolchain, because those targets are not in upstream
Rust:

```sh
cargo install espup
espup install
# then source the export script espup prints
```

Then change the chip feature in `firmware/esp32c3/Cargo.toml` (`esp32s3`
instead of `esp32c3`, on all three `esp-*` crates) and set the target in
`.cargo/config.toml` to `xtensa-esp32s3-none-elf`.

This path is **not verified here** — only the RISC-V build was compiled and
measured. The S3 is the more interesting target if you want to add a display,
since it has PSRAM and more pins.

## Ideas worth building

- **A display.** The report renderer already writes to a fixed buffer; pointing
  it at an SSD1306 or a small TFT instead of a UART is a contained change.
- **Wi-Fi catalog updates.** Fetch a fresh `.ocb` into a spare OTA partition.
  The format is designed for this: a reader validates the header and refuses
  a version it does not understand rather than misreading it.
- **A captive portal.** Serve the console as a web page so a phone can drive
  it with no serial terminal.
- **QR input.** Encode a hardware profile as a QR code; point the board's
  camera at a laptop screen and get an answer without typing.
