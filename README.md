# vuiocodecaac: High-Performance Pure Rust MPEG AAC Audio Codec Suite

N.B.! This is Rust fork of original C project https://github.com/ittiam-systems/libxaac

`vuiocodecaac` is a 100% pure, memory-safe, idiomatic Rust implementation of the MPEG AAC audio codec suite, providing bit-exact output, extreme throughput, SIMD auto-vectorization, Rayon multi-threading, and zero-allocation aligned buffer architectures.

---

## Features

- **Formats** (see [`text/plan.txt`](text/plan.txt) for the file-by-file audit
  against the C reference — it marks every tool done, partial or missing):
  - **MPEG-2 / MPEG-4 AAC-LC** (Low Complexity) — decode and encode
  - **HE-AAC v1** (Spectral Band Replication / SBR) — **decode only**
  - **HE-AAC v2** (Parametric Stereo / PS) — **decode only**
  - **MPEG-D USAC** — Frequency Domain core and the ACELP speech core;
    TCX, FAC and ISO USAC framing are not implemented yet
  - **DRC** — the legacy `dynamic_range_info()` element and BS.1770 loudness
    measurement; MPEG-D uniDRC is not implemented
  - **Not implemented**: MPEG Surround, SBR/PS encode, AAC-LD / AAC-ELD,
    error resilience (HCR/RVLC) and error concealment. These return an
    explicit `Unimplemented` error rather than approximating.
- **Container Multiplexers & Syntaxes**:
  - **ADTS** (Audio Data Transport Stream with CRC-16)
  - **ADIF** (Audio Data Interchange Format)
  - **LATM / LOAS** (Low-overhead Audio Transport Multiplex)
  - **RAW** (Direct AudioSpecificConfig payloads)
- **Performance & SIMD**:
  - 64-byte CPU cache-line aligned audio buffers (`AVec<T>` matching AVX-512 / NEON)
  - Multi-threaded multi-channel processing via **Rayon**
  - Optimized mixed-radix FFT / IMDCT transform pipelines
- **Memory Safe & Robust**:
  - `#![forbid(unsafe_code)]` enforced across all core modules

---

## License
Apache 2.0 or MIT

## Attributions

This product includes algorithmic designs and software derived from libxaac,
originally developed by Ittiam Systems Pvt. Ltd. and licensed under the 
Apache License, Version 2.0.

Original Project: https://github.com/ittiam-systems/libxaac
Copyright (c) Ittiam Systems Pvt. Ltd.