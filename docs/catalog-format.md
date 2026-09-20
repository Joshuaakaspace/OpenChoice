# The `.ocb` catalog format

A catalog is a header, a run of fixed-width records, and a deduplicated string
table. There is no compression, no index, and no framing — the point is that a
reader can `mmap` it or point at flash and start answering questions without
parsing anything.

After filtering, 3,923 models pack to **258 KB** — about 67 bytes each against
roughly 890 bytes each in the source JSON. That ratio is what makes the whole
project possible.

All integers are **little-endian**. All offsets are from the start of the file.

## Header — 48 bytes

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 4 | magic | `"OCB1"` |
| 4 | 2 | version | currently `2`; a reader rejects anything else |
| 6 | 2 | record_size | currently `32`; rejected if it differs |
| 8 | 4 | record_count | |
| 12 | 4 | records_offset | |
| 16 | 4 | strings_offset | |
| 20 | 4 | strings_length | |
| 24 | 4 | measurements_offset | |
| 28 | 4 | measurement_count | zero when the catalog carries no benchmarks |
| 32 | 4 | calibrations_offset | |
| 36 | 4 | calibration_count | |
| 40 | 4 | flags | reserved, zero |
| 44 | 4 | checksum | reserved, zero |

Version 2 added the two measurement sections. Version 1 files are refused
outright rather than read with the new fields defaulted — a reader that
guesses at a layout it does not know is how a device ends up confidently
wrong.

`record_size` is in the header and checked rather than assumed. A future
version that widens the record can be detected and refused cleanly instead of
being misread field by field — which on a device with no console is the
difference between a clear error and inexplicable nonsense.

## Record — 32 bytes

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 4 | name_offset | into the string table |
| 4 | 4 | params_m | total parameters, millions |
| 8 | 4 | active_params_m | per-token parameters; `0` means "same as total" |
| 12 | 4 | context_length | tokens |
| 16 | 2 | hidden_size | |
| 18 | 2 | num_layers | |
| 20 | 2 | num_kv_heads | |
| 22 | 2 | head_dim | |
| 24 | 2 | vocab_k | vocabulary size / 1000 |
| 26 | 2 | flags | see below |
| 28 | 1 | quant_mask | bit N = `Quant` discriminant N; `0` means "unknown, assume the standard ladder" |
| 29 | 1 | use_case | `UseCase` discriminant |
| 30 | 1 | family | index into the packer's family table |
| 31 | 1 | quality_prior | 0-255 |

### Flags

| Bit | Name | Meaning |
|---:|---|---|
| 0 | `MOE` | sparse; only `active_params_m` move per token |
| 1 | `VISION` | accepts images |
| 2 | `EMBEDDING` | embedding model, not a generator |
| 3 | `ARCH_METADATA` | layer and head counts are real, not estimated |
| 4 | `INSTRUCT` | instruction-tuned |
| 5 | `GGUF` | a GGUF conversion is known to exist |

`ARCH_METADATA` is the one that matters for honesty. Without it the engine
infers layer and head counts from a parameter size class, and every KV-cache
figure derived that way is tagged `KvSource::Estimated` so a caller can never
present an inference as a measurement.

`quality_prior` is derived from HuggingFace download and like counts. It is
**popularity, not measured quality** — it is bounded to ±15 points in the
scorer precisely so it can never override parameter count or a bad fit.

## String table

NUL-terminated UTF-8, deduplicated by content. A `name_offset` past the end of
the table yields an empty string rather than a panic: a corrupt catalog should
degrade to useless output, not take down a device that cannot report a
backtrace.

## Measurement — 16 bytes

Real benchmark results, sorted by `(hw_key, model_index)` so a lookup is a
binary search with no allocation.

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 4 | hw_key | FNV-1a of the machine name, alphanumerics only, lowercased |
| 4 | 4 | model_index | position in the record section |
| 8 | 2 | tps_x10 | measured decode throughput x10 |
| 10 | 2 | ttft_ms | time to first token in that run; 0 if not recorded |
| 12 | 1 | quant | `Quant` discriminant the run actually used |
| 13 | 1 | runs | how many submitted runs were aggregated |
| 14 | 1 | provider | 1 llama.cpp, 2 ollama, 3 mlx, 4 vllm |
| 15 | 1 | flags | reserved, zero |

Measurements are keyed by **catalog index, not by a name hash**. Reconciling
`qwen3:8b` with `Qwen/Qwen3-8B` is fuzzy work; doing it once in the packer
means it can be counted, printed, and corrected. Doing it on the device would
mean every reader re-deriving the same guess with no way to audit it.

The `quant` field is what lets a Q4 benchmark inform a Q8 row honestly:
decode is bandwidth-bound, so the engine rescales by the ratio of bits per
weight and downgrades the label from `measured` to `measured-adjusted`.

## Calibration — 8 bytes

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 4 | hw_key | same hash as above |
| 4 | 2 | factor_x1000 | multiplier on the roofline estimate |
| 6 | 1 | samples | measurements the factor was derived from |
| 7 | 1 | flags | reserved, zero |

Derived at pack time as the median of `measured / predicted` across every
measurement on that machine, using this engine to produce the prediction — so
a factor always describes the formula that is actually shipping. Ratios
outside 0.05–5.0 are discarded as misidentified pairings rather than allowed
to poison the median, and a machine with fewer than two samples gets no factor
at all: one measurement is an anecdote.

## Building one

```sh
# everything worth keeping
openchoice-catalog build -i hf_models.json -o catalog/openchoice.ocb

# with real measurements folded in
openchoice-catalog build -i hf_models.json --community ./community -o out.ocb

# see which benchmark names matched no catalog model
openchoice-catalog build -i hf_models.json --community ./community \
  -o out.ocb --show-unmatched

# a small one for a tight flash budget, most-downloaded first
openchoice-catalog build -i hf_models.json -o tiny.ocb --top 512

# what did I actually get?
openchoice-catalog inspect catalog/openchoice.ocb
```

The packer drops entries below 1M parameters or below `--min-downloads`
(default 1000), and by default keeps only text, vision-text, and embedding
pipelines. The source catalog carries a long tail of near-empty test uploads;
they cost flash and can only dilute a ranking.

Every `build` re-parses its own output before reporting success. A catalog
that does not round-trip is worse than no catalog, and the packer is the last
place that can catch it before a device tries to boot on it.

## Sizing

| Models | File size | Notes |
|---:|---:|---|
| 512 | ~35 KB | fits anywhere, including an ESP8266-class part |
| 1,500 | 98 KB | the firmware default, with 309 measurements |
| 3,923 | 258 KB | everything that survives default filtering, 358 measurements |

Roughly 67 bytes per model all-in: the 32-byte record plus its share of the
string table. The measurement and calibration sections add about 5.7 KB
regardless of how many models are kept, since they scale with how many
benchmarks exist, not with catalog size.
