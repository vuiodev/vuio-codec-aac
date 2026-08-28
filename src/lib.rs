//! # vuiocodecaac
//!
//! An MPEG-4 AAC-LC decoder and encoder, ported from the reference C
//! implementation [libxaac](https://github.com/ittiam-systems/libxaac).
//!
//! ## Scope
//!
//! The decoder implements AAC-LC: all four window sequences, both window shapes,
//! the full spectral and scalefactor Huffman codebooks, section data, temporal
//! noise shaping, mid/side and intensity stereo, noise substitution and pulse
//! data, framed in ADTS. It is verified against the C reference; see the README
//! for the measured figures.
//!
//! Spectral Band Replication is implemented: an HE-AAC stream decodes at its full
//! nominal sample rate, with the replicated range reconstructed from the transmitted
//! envelopes, noise floors and added sinusoids. Parametric Stereo is implemented, so
//! HE-AAC v2 decodes to stereo. The encoder emits conformant AAC-LC with a real
//! psychoacoustic model, block switching, TNS and mid/side.
//!
//! ## What is not implemented
//!
//! `text/plan.txt` is the authoritative audit against the C reference, file by file.
//! In short:
//!
//! * **MPEG Surround** — not implemented; [`decoder::mps`] returns
//!   [`error::Error::Unimplemented`] rather than approximating.
//! * **SBR and PS encode** — not implemented, same treatment. The *decode* side of
//!   both is real, so this crate reads HE-AAC v1/v2 but cannot write it.
//! * **USAC / xHE-AAC** — the Frequency Domain core and the ACELP speech core are
//!   implemented; TCX, FAC and the ISO `UsacConfig()`/`UsacFrame()` framing are not,
//!   so USAC currently round-trips only through this crate's own container.
//! * **MPEG-D uniDRC** — only the legacy `dynamic_range_info()` element is handled.
//! * **AAC-LD / ELD**, error resilience (HCR, RVLC), error concealment, LTP and
//!   coupling-channel gains — not implemented; where the syntax must still be
//!   traversed to stay bit-aligned, it is parsed and skipped.
//!
//! Nothing in this crate returns a fabricated result for an unimplemented tool: the
//! rule is an error, never a plausible-looking wrong answer.
//!
//! ## Quickstart: Decoding an AAC Stream
//!
//! ```no_run
//! use vuiocodecaac::prelude::*;
//!
//! let mut decoder = Decoder::new_default();
//! let adts_packet = [0xFF, 0xF1, 0x50, 0x80, 0x02, 0x1F, 0xFC /* ... */];
//! let pcm_frame = decoder.decode_frame(&adts_packet).unwrap();
//!
//! println!("Decoded {} channels x {} samples", pcm_frame.channels(), pcm_frame.samples_per_channel());
//! ```
//!
//! ## Quickstart: Encoding Audio to AAC
//!
//! ```no_run
//! use vuiocodecaac::prelude::*;
//!
//! let config = EncoderConfig::default();
//! let mut encoder = Encoder::new(config).unwrap();
//! let pcm_input = AudioBuffer::<i16>::new(2, 1024);
//! let adts_packet = encoder.encode_frame(&pcm_input).unwrap();
//! ```

// `unsafe` is denied crate-wide and allowed in exactly one place: `dsp::simd`,
// which wraps SIMD intrinsics. Those intrinsics are `unsafe` because they can
// require CPU features the machine may lack and because their load/store forms take
// raw pointers; that module documents how both obligations are discharged, and every
// kernel it contains is checked against a scalar reference in its tests.
#![deny(unsafe_code)]

pub mod bitstream;
pub mod buffer;
pub mod decoder;
pub mod dsp;
pub mod encoder;
pub mod error;
pub mod syntax;
pub mod tables;
pub mod types;

/// Convenient re-exports for general codec usage.
pub mod prelude {
    pub use crate::bitstream::{BitReader, BitWriter};
    pub use crate::buffer::AudioBuffer;
    pub use crate::decoder::Decoder;
    pub use crate::encoder::{Encoder, EncoderConfig};
    pub use crate::error::{Error, Result};
    pub use crate::syntax::{AdtsHeader, AudioSpecificConfig};
    pub use crate::types::{
        AudioObjectType, BitstreamFormat, ChannelConfiguration, FrameLength, SamplingRate,
        WindowSequence, WindowShape,
    };
}
