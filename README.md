# vuiocodecaac

A Rust implementation of the MPEG-4 AAC-LC decoder and encoder, ported from the
reference C implementation [libxaac](https://github.com/ittiam-systems/libxaac)
(Ittiam Systems, Apache 2.0).

## Status

The decoder implements **AAC-LC** and is verified against the C reference; the
encoder produces conformant AAC-LC that libxaac, ffmpeg and this decoder all
agree on. The other codec families in libxaac are **not implemented** — see the
coverage table below before depending on this crate.

### Decoder

| Tool | Status |
| :--- | :--- |
| AAC-LC: all four window sequences, sine and KBD shapes | Implemented |
| Huffman codebooks 1–11 with escape coding, scalefactor codebook | Implemented, tables ported verbatim from the C ROM |
| Section data, scalefactors, window grouping | Implemented |
| Temporal Noise Shaping (TNS) | Implemented |
| Mid/side and intensity stereo | Implemented |
| Perceptual Noise Substitution (PNS) | Implemented |
| Pulse data | Implemented |
| ADTS framing | Implemented |
| LATM/LOAS, ADIF | Parsers exist in `syntax::`, not wired into the decoder |
| Coupling channel elements (CCE) | Parsed for bit alignment; gains not applied |
| LTP, main-profile prediction, SSR gain control | Parsed and skipped |
| SBR / eSBR (HE-AAC) | **Not implemented** — HE-AAC decodes to its core at half rate |
| Parametric Stereo (HE-AAC v2) | **Not implemented** — decodes to the mono downmix |
| MPEG Surround, USAC, DRC | **Not implemented** — placeholder modules only |
| RVLC, HCR, error concealment | **Not implemented** — corrupt frames are dropped |

### Encoder

Emits conformant AAC-LC: long windows, sine shape, rate control by bisection on a
single scalefactor. It has **no psychoacoustic model, no block switching, no
mid/side stereo and no TNS**, so at a given bitrate its quality is well below
ffmpeg's or libxaac's. It is correct, not competitive.

## Verification

`tools/verify_corpus.sh` decodes a corpus with this decoder, the C reference and
ffmpeg, and compares sample by sample. Across 16 streams (8 signals × 2 encoders,
16–48 kHz, mono and stereo):

- **52–102 dB SNR** against the C reference, with ~69 % of samples bit-identical
  and a maximum error of 2 LSB over 5.3 M samples.
- Every remaining divergence is attributable to **Perceptual Noise Substitution**,
  where the bitstream carries only a band's energy and each decoder synthesises its
  own noise. On streams where this decoder disagrees with ffmpeg, the C reference
  disagrees with ffmpeg by the same margin to within 0.02 dB.

Entropy decoding is bit-exact by construction: the codebook tables are generated
from the C ROM by `tools/extract_rom.py` and the search is the reference's own
algorithm. Downstream stages are floating point, so they match the reference's
fixed-point pipeline to within float rounding rather than exactly.

```sh
cargo test --release                        # 146 tests
./tools/verify_corpus.sh <corpus> <xaacdec> # three-way decoder comparison
cargo bench --bench codec_bench             # per-kernel timings
```

## Performance

Decoding a two-minute 128 kbps stereo stream on aarch64 (Apple Silicon), decode
time only:

| Configuration | Throughput | Per frame |
| :--- | ---: | ---: |
| libxaac C reference | 800× realtime | — |
| This crate, single thread | 1,299× realtime | 17.9 µs |
| This crate, 8 threads | 4,574× realtime | 5.1 µs |

The inverse transform is a quarter-length complex FFT (replacing an O(N²) matrix
product), with radix-4 butterflies vectorized through NEON on aarch64 and AVX/FMA
on x86-64, and a scalar fallback elsewhere.

Parallel decoding is exact rather than approximate: frame dependencies reach back
exactly one frame, so each worker primes on the preceding frame and discards it.
`tests/parallel_decode.rs` asserts byte-identical output against a single-threaded
run.

```sh
aacdec in.aac out.wav --parallel     # threaded decode
aacdec in.aac /dev/null --repeat 5   # decode-only benchmark
AAC_TRACE=1 aacdec in.aac /dev/null  # per-frame tools and PNS energy share
```

### On `unsafe`

`unsafe` is denied crate-wide with one audited exception, `dsp::simd`, which wraps
SIMD intrinsics. NEON is architecturally guaranteed on aarch64; the x86-64 path is
gated on runtime feature detection; every load and store is bounds-checked against
slice lengths before a pointer is formed; and each kernel is checked against a
scalar reference across every span the FFT plans use.

### Noise substitution and reproducibility

PNS noise is synthesised, not transmitted, so its samples are decoder-specific by
design. Two modes are available (`Decoder::set_noise_mode`):

- `Sequential` (default) threads one generator through the stream as the reference
  does, which is what the fidelity figures above measure.
- `PerFrame` seeds from the frame position, so seeking and parallel decoding are
  byte-exact. The batch APIs in `decoder::batch` use this mode.

## Porting tools

The ported tables are generated from the C source rather than transcribed, so they
can be regenerated after a reference update:

- `tools/extract_rom.py` — Huffman codebooks (`src/tables/huffman_rom.rs`)
- `tools/extract_sfb.py` — scalefactor-band and TNS tables (`src/tables/sfb.rs`)

## License

Apache 2.0 or MIT.

## Attribution

This product includes algorithmic designs and software derived from libxaac,
originally developed by Ittiam Systems Pvt. Ltd. and licensed under the Apache
License, Version 2.0.

Original project: https://github.com/ittiam-systems/libxaac
Copyright (c) Ittiam Systems Pvt. Ltd.
