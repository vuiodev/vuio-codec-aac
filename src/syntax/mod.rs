//! MPEG AAC Transport and Syntax Demuxers
//!
//! Implements container framing, multiplexing, and header parsing for:
//! - ADTS (Audio Data Transport Stream)
//! - ADIF (Audio Data Interchange Format)
//! - LOAS / LATM (Low-overhead Audio Stream / Multiplex)
//! - AudioSpecificConfig (ASC) and ProgramConfigElement (PCE)
//! - RAW AAC elementary streams

pub mod adif;
pub mod adts;
pub mod asc;
pub mod latm;
pub mod pce;
pub mod raw;

pub use adif::AdifHeader;
pub use adts::AdtsHeader;
pub use asc::AudioSpecificConfig;
pub use latm::{AudioMuxElement, StreamMuxConfig};
pub use pce::ProgramConfigElement;
pub use raw::RawAudioFrame;
