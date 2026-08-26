//! MPEG-4 AAC Core Decoding Subsystem (LC, LD, ELD)

pub mod channel;
pub mod dequant;
pub mod huffman;
pub mod pns;
pub mod stereo;
pub mod tns;

pub use channel::{ElementType, IcsInfo, SingleChannelElement, ChannelPairElement};
pub use dequant::inverse_quantize;
pub use huffman::decode_spectral_band;
pub use pns::PnsGenerator;
pub use stereo::{apply_intensity_stereo, apply_ms_stereo};
pub use tns::TnsFilter;
