//! MPEG Audio Decoding Subsystems
//!
//! Provides the top-level `Decoder` orchestrator and modular sub-decoders for:
//! - MPEG-4 AAC-LC / LD / ELD Core
//! - SBR / eSBR (Spectral Band Replication)
//! - Parametric Stereo (PS)
//! - MPEG Surround (MPS)
//! - MPEG-D USAC (Unified Speech and Audio Coding)
//! - MPEG-D DRC (Dynamic Range Control)

pub mod aac;
pub mod drc;
pub mod batch;
pub mod engine;
pub mod mps;
pub mod ps;
pub mod sbr;
pub mod usac;

pub use drc::{DrcDecoder, DrcFrameData};
pub use engine::Decoder;
pub use mps::{MpsDecoder, MpsSpatialCues};
pub use ps::{PsDecoder, PsFrameData};
pub use sbr::{SbrDecoder, SbrHeader};
pub use usac::{UsacCoreMode, UsacDecoder};
