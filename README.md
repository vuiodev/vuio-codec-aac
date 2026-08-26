# XAAC: High-Performance Pure Rust MPEG AAC Audio Codec Suite

`xaac` is a 100% pure, memory-safe, idiomatic Rust 2024 implementation of the MPEG AAC audio codec suite, providing bit-exact output, extreme throughput, SIMD auto-vectorization, Rayon multi-threading, and zero-allocation aligned buffer architectures.

---

## Features

- **Full Format Suite**:
  - **MPEG-2 / MPEG-4 AAC-LC** (Low Complexity)
  - **HE-AAC v1** (Spectral Band Replication / SBR)
  - **HE-AAC v2** (Parametric Stereo / PS)
  - **AAC-LD / AAC-ELD** (Low Delay / Enhanced Low Delay communication)
  - **MPEG-D USAC** (Unified Speech and Audio Coding: ACELP / TCX / FD)
  - **MPEG-D DRC / UniDRC** (Dynamic Range Control & BS.1770 Loudness Normalization)
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