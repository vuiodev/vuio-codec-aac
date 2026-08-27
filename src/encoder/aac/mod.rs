//! MPEG-4 AAC Core Encoding Subsystem

pub mod bitstream;
pub mod huffman;
pub mod block_switch;
pub mod psycho;
pub mod rate;
pub mod quant;

pub use bitstream::{
    finalize_adts_frame, write_cpe, write_fill_sbr, write_lfe, write_multichannel_elements,
    write_sce,
};
pub use block_switch::BlockSwitching;
pub use psycho::PsychoacousticModel;
pub use quant::{BandChoice, choose_codebook, quantize_band, write_band};
