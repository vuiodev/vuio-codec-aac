//! MPEG Surround (MPS) multi-channel spatial audio decoder — **not implemented**.
//!
//! MPEG Surround (ISO/IEC 23003-1) reconstructs a multichannel signal from a
//! transmitted downmix plus spatial cues: Channel Level Differences (CLD),
//! Inter-channel Coherence (ICC), Channel Prediction Coefficients (CPC) and
//! phase cues. The real decoder parses a spatial-specific config, runs the
//! downmix through a hybrid QMF filterbank, builds per-band M1/M2 matrices for
//! the signalled tree configuration, synthesises decorrelated channels, applies
//! temporal shaping, and inverts the filterbank.
//!
//! None of that is here. This module exists so the gap has a name and an
//! address, not because any of it works.
//!
//! # Why this is an error and not an approximation
//!
//! An earlier revision of this file shipped a fixed matrix upmix — it computed
//! `c = 0.5*(l+r)`, derived surrounds by subtraction, and returned that as
//! "MPEG Surround". It read no bitstream and used none of the transmitted cues,
//! so it produced confident, plausible, wrong output for every input, and its
//! doc comment claimed otherwise. That is the failure mode this port treats as
//! worse than a missing feature, so the fabricated maths is gone and
//! [`MpsDecoder::decode`] refuses instead.
//!
//! If you want a matrix upmix, write one at the call site where its limits are
//! visible. Do not let it wear this module's name.
//!
//! # Porting this for real
//!
//! `text/plan.txt` phase 9 has the breakdown: ~35,300 lines of C across
//! `c/libxaac/decoder/ixheaacd_mps_*.c`, the largest single subsystem in the
//! reference. Start at `ixheaacd_mps_parse.c` and `ixheaacd_mps_bitdec.c` for
//! the config and payload, then the filterbanks, then the M1/M2 matrices.

use crate::error::{Error, Result};

/// MPEG Surround spatial parameters for 5.1 / 7.1 surround synthesis.
///
/// Kept as the shape the real decoder will fill in; nothing populates it today.
#[derive(Debug, Clone, Default)]
pub struct MpsSpatialCues {
    /// Channel Level Differences, per parameter band.
    pub cld: Vec<f32>,
    /// Inter-channel Coherence, per parameter band.
    pub icc: Vec<f32>,
    /// Channel Prediction Coefficients, per parameter band.
    pub cpc: Vec<f32>,
}

/// MPEG Surround spatial audio renderer.
pub struct MpsDecoder {
    tree_config: u8,
}

impl MpsDecoder {
    /// Create a decoder for a signalled tree configuration (0 = 5.1, 1 = 7.1,
    /// 2 = binaural).
    pub fn new(tree_config: u8) -> Self {
        Self { tree_config }
    }

    /// The tree configuration this decoder was constructed for.
    pub fn tree_config(&self) -> u8 {
        self.tree_config
    }

    /// Render a downmix to its multichannel output.
    ///
    /// Always returns [`Error::Unimplemented`]. See this module's documentation
    /// for why it refuses rather than approximating.
    pub fn decode(
        &self,
        _downmix: &[&[f32]],
        _cues: &MpsSpatialCues,
        _out: &mut [Vec<f32>],
    ) -> Result<()> {
        Err(Error::Unimplemented {
            tool: "MPEG Surround decode",
            detail: "text/plan.txt phase 9; c/libxaac/decoder/ixheaacd_mps_*.c",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contract is that this refuses. If someone reintroduces a fabricated
    /// upmix, this test is what fails.
    #[test]
    fn decoding_is_refused_rather_than_approximated() {
        let mps = MpsDecoder::new(0);
        let left = vec![1.0f32; 512];
        let right = vec![0.5f32; 512];
        let mut out = vec![vec![0.0f32; 512]; 6];
        let err = mps.decode(&[&left, &right], &MpsSpatialCues::default(), &mut out).unwrap_err();
        assert!(matches!(err, Error::Unimplemented { .. }));
        assert!(out.iter().all(|ch| ch.iter().all(|s| *s == 0.0)), "output must be untouched");
    }
}
