//! MPEG Audio Encoding Subsystems
//!
//! Provides the top-level `Encoder` orchestrator and modular encoders for:
//! - MPEG-4 AAC-LC / LD / ELD Core
//! - SBR (Spectral Band Replication)
//! - Parametric Stereo (PS)
//! - MPEG-D USAC (Unified Speech and Audio Coding)
//! - MPEG-D DRC (Dynamic Range Control)

pub mod aac;
pub mod drc;
pub mod engine;
pub mod ps;
pub mod sbr;
pub mod usac;

pub use drc::DrcEncoder;
pub use engine::{Encoder, EncoderConfig};
pub use ps::PsEncoder;
pub use sbr::SbrEncoder;
pub use usac::UsacEncoder;
