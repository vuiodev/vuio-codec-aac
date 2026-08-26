# XAAC: High-Performance Pure Rust MPEG AAC Audio Codec Suite

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/rust-2024%20edition-orange.svg)](https://www.rust-lang.org/)
[![Safety](https://img.shields.io/badge/unsafe-forbidden-success.svg)](#memory-safety)

`xaac` is a 100% pure, memory-safe, idiomatic Rust 2024 implementation of the MPEG AAC audio codec suite, providing bit-exact output, extreme throughput, SIMD auto-vectorization, Rayon multi-threading, and zero-allocation aligned buffer architectures.

---

## Features

- **Full Format Suite**:
  - **MPEG-2 / MPEG-4 AAC-LC** (Low Complexity)
  - **HE-AAC v1** (Spectral Band Replication / SBR)
  - **HE-AAC v2** (Parametric Stereo / PS)
  - **AAC-LD / AAC-ELD** (Low Delay / Enhanced Low Delay communication)
  - **MPEG-D USAC** (Unified Speech and Audio Coding)
  - **MPEG-D DRC / UniDRC** (Dynamic Range Control & Loudness metadata)
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
  - Typed error hierarchy with `thiserror 2.0`

---

## Installation & Usage

Add `xaac` to your `Cargo.toml`:

```toml
[dependencies]
xaac = "0.1.0"
```

### Decoding AAC to PCM

```rust
use xaac::prelude::*;

fn main() -> Result<()> {
    let mut decoder = Decoder::new_default();
    
    // Read raw or ADTS-framed AAC bytes
    let adts_frame = [0xFF, 0xF1, 0x50, 0x80, 0x02, 0x1F, 0xFC /* ... */];
    let pcm = decoder.decode_frame(&adts_frame)?;

    println!("Decoded {} channels, {} samples", pcm.channels(), pcm.samples_per_channel());
    Ok(())
}
```

### Encoding PCM to AAC

```rust
use xaac::prelude::*;

fn main() -> Result<()> {
    let config = EncoderConfig {
        audio_object_type: AudioObjectType::AacLc,
        sampling_rate: SamplingRate::Hz44100,
        channel_config: ChannelConfiguration::Stereo,
        bitrate_bps: 128_000,
        frame_length: FrameLength::Samples1024,
    };

    let mut encoder = Encoder::new(config)?;
    let pcm_input = AudioBuffer::<i16>::new(2, 1024);
    
    let adts_packet = encoder.encode_frame(&pcm_input)?;
    println!("Encoded ADTS frame: {} bytes", adts_packet.len());
    Ok(())
}
```

---

## CLI Tools

### `aacdec`: High-Speed Audio Decoder

```bash
cargo run --release --bin aacdec -- input.aac output.wav
```

### `aacenc`: Audio Encoder

```bash
cargo run --release --bin aacenc -- input.wav output.aac --bitrate 192000
```

---

## Benchmarks

Run criterion benchmarks:

```bash
cargo bench
```

---

## License
Apache 2.0 or MIT