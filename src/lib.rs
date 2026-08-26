//! # vuiocodecaac: High-Performance, Bit-Exact Pure Rust Audio Codec
//!
//! `vuiocodecaac` is a memory-safe, 100% idiomatic Rust 2024 implementation of the MPEG AAC,
//! HE-AAC v1/v2 (SBR + PS), AAC-ELD, MPEG-D USAC, and MPEG-D DRC audio codec suite.
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

#![forbid(unsafe_code)]

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
