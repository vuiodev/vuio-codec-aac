//! Parametric Stereo (PS) encoder — **not implemented**.
//!
//! A real PS encoder takes a stereo pair into the hybrid QMF domain, extracts
//! Inter-channel Intensity Differences, Inter-channel Coherence and (optionally)
//! phase differences *per hybrid subband per parameter set*, quantises them
//! against the standard's IID/ICC tables, delta-codes them across time or
//! frequency, and writes a `ps_data()` payload alongside a mono downmix.
//!
//! None of that is here.
//!
//! # Why this is an error and not an approximation
//!
//! An earlier revision computed one broadband IID and one broadband ICC from
//! time-domain energies, produced a `0.5*(l+r)` downmix, and returned them in a
//! struct. Real PS parameters are per band and per parameter set; a single
//! broadband pair cannot represent the stereo image the decoder expects, and no
//! payload was written at all, so nothing could consume it. It was never called
//! by [`crate::encoder::engine`].
//!
//! # Porting this for real
//!
//! `text/plan.txt` phase 4.6, ~1,600 lines of C:
//! `ixheaace_ps_enc.c`, `ixheaace_ps_bitenc.c`, `ixheaace_ps_enc_init.c` and the
//! encoder-side hybrid filterbank in `ixheaace_hybrid.c`. The decode side —
//! [`crate::decoder::ps`], including its hybrid filterbank and decorrelator —
//! is implemented and tested, so a round trip is the natural oracle. PS also
//! depends on SBR encode (phase 4.1-4.5) being in place first, since PS travels
//! inside the SBR extension payload.

use crate::error::{Error, Result};

/// Parameters one frame of parametric stereo carries.
///
/// Kept as the shape the real encoder will fill in. `iid_indices` and
/// `icc_indices` are per hybrid band, not scalars.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PsFrameData {
    pub enable_iid: bool,
    pub enable_icc: bool,
    pub enable_ipd: bool,
    pub iid_indices: Vec<i8>,
    pub icc_indices: Vec<u8>,
}

/// PS encoder instance.
#[derive(Debug, Default, Clone, Copy)]
pub struct PsEncoder;

impl PsEncoder {
    pub fn new() -> Self {
        Self
    }

    /// Extract PS parameters and produce the mono downmix that accompanies them.
    ///
    /// Always returns [`Error::Unimplemented`]. See this module's documentation
    /// for why it refuses rather than returning broadband stand-ins.
    pub fn encode_stereo(
        &self,
        _left: &[f32],
        _right: &[f32],
        _mono_downmix: &mut [f32],
    ) -> Result<PsFrameData> {
        Err(Error::Unimplemented {
            tool: "Parametric Stereo encode",
            detail: "text/plan.txt phase 4.6; c/libxaac/encoder/ixheaace_ps_*.c",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_is_refused_rather_than_returning_broadband_stand_ins() {
        let enc = PsEncoder::new();
        let mut downmix = vec![0.0f32; 512];
        let err =
            enc.encode_stereo(&vec![1.0f32; 512], &vec![0.5f32; 512], &mut downmix).unwrap_err();
        assert!(matches!(err, Error::Unimplemented { .. }));
        assert!(downmix.iter().all(|s| *s == 0.0), "output must be untouched");
    }
}
