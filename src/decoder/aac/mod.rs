//! MPEG-4 AAC Core Decoding Subsystem (LC, LD, ELD)

pub mod channel;
pub mod dequant;
pub mod downmix;
pub mod huffman;
pub mod ics;
pub mod pns;
pub mod stereo;
pub mod tns;

pub use channel::{ElementType, IcsInfo, SingleChannelElement, ChannelPairElement};
pub use dequant::{inverse_quantize, inverse_quantize_channel};
pub use huffman::decode_spectral_band;
pub use ics::{ChannelData, IcsInfo as IcsLayout, TnsData, decode_ics, deinterleave};
pub use pns::{NoiseRng, apply_pns};
pub use stereo::{MsMask, apply_intensity_stereo, apply_ms_stereo};
pub use tns::{apply_tns, ar_filter, ma_filter, parcor_to_lpc};
