//! LSF dequantization for USAC's LPD speech-coding mode.
//!
//! Pairs with [`crate::encoder::usac::lsf`]; see [`crate::tables::usac_lsf`] for the
//! shared conversion math and codebook, and for what is and is not covered by this
//! minimal (codebook-only, no lattice refinement) quantizer.

use crate::tables::usac_lsf::{DICO_LSF_ABS_8B, LPC_ORDER, enforce_lsf_stability};

/// Reconstruct the LSF vector a transmitted codebook index names: a direct
/// table lookup, since this module's quantizer has no residual stage to add
/// back on top (see [`crate::encoder::usac::lsf::quantize_lsf_abs`]) —
/// `ixheaacd_avq_first_approx_abs` minus the AVQ refinement this codebase does
/// not yet implement.
///
/// [`enforce_lsf_stability`] is applied defensively even though every real
/// codeword in [`DICO_LSF_ABS_8B`] is already ordered and stable by
/// construction: it is what protects a caller from an out-of-range or
/// corrupted index (e.g. from a damaged bitstream) turning into an unstable
/// LPC synthesis filter instead of a clamped, still-ordered one.
pub fn dequantize_lsf_abs(index: u8) -> [f32; LPC_ORDER] {
    let mut lsf = DICO_LSF_ABS_8B[index as usize];
    enforce_lsf_stability(&mut lsf);
    lsf
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every real codeword must already be a valid, ordered, minimum-spaced
    /// LSF set -- if the table were ever ported with a transcription error
    /// this would be the cheapest place to catch it.
    #[test]
    fn every_codeword_is_already_ordered_and_stable() {
        for index in 0..=255u8 {
            let lsf = dequantize_lsf_abs(index);
            let raw = DICO_LSF_ABS_8B[index as usize];
            assert_eq!(lsf, raw, "a real codeword must never need clamping: index {index}");
            for w in lsf.windows(2) {
                assert!(w[1] > w[0], "codeword {index} is not strictly ordered: {lsf:?}");
            }
        }
    }
}
