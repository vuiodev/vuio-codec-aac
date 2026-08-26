//! MPEG-4 AAC Core Encoding Subsystem

pub mod bitstream;
pub mod block_switch;
pub mod psycho;
pub mod quant;

pub use bitstream::{
    finalize_adts_frame, write_cpe, write_fill_sbr, write_lfe, write_multichannel_elements,
    write_sce,
};
pub use block_switch::BlockSwitching;
pub use psycho::PsychoacousticModel;
pub use quant::{estimate_global_gain, quantize_band};
