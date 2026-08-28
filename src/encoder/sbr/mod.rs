//! Spectral Band Replication (SBR) encoder — **not implemented**.
//!
//! A real SBR encoder runs the input through a QMF analysis bank, decides an
//! envelope time/frequency grid from where transients fall, estimates envelope
//! energies, noise floors, inverse-filtering levels and missing harmonics over
//! that grid, then delta-codes the lot into an `sbr_extension_data()` payload
//! the decoder in [`crate::decoder::sbr`] can read.
//!
//! None of that is here.
//!
//! # Why this is an error and not an approximation
//!
//! An earlier revision ran a genuine QMF analysis, discarded everything except
//! a scalar energy average, and returned five hard-coded bytes
//! (`[0x00, 0x2A, 0x44, 0x80, quantised_energy]`) described as an SBR payload.
//! No decoder can read that — not this crate's own, which is the point. The
//! only reason the encoder still emitted valid streams is that
//! [`crate::encoder::engine`] never called it.
//!
//! Returning a wrong payload is worse than returning nothing, because a caller
//! who wires it up gets a stream that looks like HE-AAC and is not, so this now
//! refuses.
//!
//! # Porting this for real
//!
//! `text/plan.txt` phase 4, ~20,000 lines of C in
//! `c/libxaac/encoder/ixheaace_sbr_*.c`. The decode side is already implemented
//! and tested here, which makes this unusually pleasant to verify: encode,
//! decode with [`crate::decoder::sbr`], and compare against the input's high
//! band. Start with `ixheaace_sbr_qmf_enc.c` and `ixheaace_sbr_frame_info_gen.c`.

use crate::error::{Error, Result};

/// SBR encoder configuration and state.
pub struct SbrEncoder {
    sample_rate: u32,
    bitrate: u32,
}

impl SbrEncoder {
    /// Create an SBR encoder for a target rate.
    pub fn new(sample_rate: u32, bitrate: u32) -> Self {
        Self { sample_rate, bitrate }
    }

    /// The core sampling rate this encoder was configured for.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The target bitrate this encoder was configured for.
    pub fn bitrate(&self) -> u32 {
        self.bitrate
    }

    /// Analyse one frame and produce its SBR extension payload.
    ///
    /// Always returns [`Error::Unimplemented`]. See this module's documentation
    /// for why it refuses rather than emitting a payload no decoder can read.
    pub fn encode_sbr_frame(&mut self, _pcm_frame: &[f32]) -> Result<Vec<u8>> {
        Err(Error::Unimplemented {
            tool: "SBR encode",
            detail: "text/plan.txt phase 4; c/libxaac/encoder/ixheaace_sbr_*.c",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_is_refused_rather_than_emitting_an_unreadable_payload() {
        let mut enc = SbrEncoder::new(44100, 128_000);
        let err = enc.encode_sbr_frame(&vec![0.25f32; 1024]).unwrap_err();
        assert!(matches!(err, Error::Unimplemented { .. }));
    }
}
