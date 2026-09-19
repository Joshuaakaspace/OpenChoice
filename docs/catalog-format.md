# The `.ocb` catalog format

A catalog is a header, a run of fixed-width records, and a deduplicated string
table. There is no compression, no index, and no framing — the point is that a
reader can `mmap` it or point at flash and start answering questions without
parsing anything.

15,049 source models pack to **252 KB** after filtering, against 13.4 MB of
equivalent JSON. That ratio is what makes the whole project possible.

All integers are **little-endian**. All offsets are from the start of the file.

## Header — 32 bytes

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 4 | magic | `"OCB1"` |
| 4 | 2 | version | currently `1`; a reader rejects anything else |
| 6 | 2 | record_size | currently `32`; rejected if it differs |
| 8 | 4 | record_count | |
| 12 | 4 | records_offset | |
| 16 | 4 | strings_offset | |
| 20 | 4 | strings_length | |
| 24 | 4 | flags | reserved, zero |
| 28 | 4 | checksum | reserved, zero |

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

## Building one

```sh
# everything worth keeping
openchoice-catalog build -i hf_models.json -o catalog/openchoice.ocb

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
| 512 | ~33 KB | fits anywhere, including an ESP8266-class part |
| 1,500 | 93 KB | the firmware default |
| 3,923 | 252 KB | everything that survives default filtering |

Roughly 64 bytes per model all-in: the 32-byte record plus its share of the
string table.
