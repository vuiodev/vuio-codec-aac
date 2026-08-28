//! TCX (Transform-Coded eXcitation): the transform half of MPEG-D USAC's
//! Linear Prediction Domain (LPD) mode, alongside the ACELP speech core in
//! [`super::acelp`].
//!
//! Ported from `c/libxaac/decoder/ixheaacd_tcx_fwd_{mdct,alcnx}.c`.
//!
//! # What TCX is doing
//!
//! Where ACELP models an excitation with two codebooks, TCX transmits it as a
//! frequency-domain spectrum — like the FD core, but shaped for the *same* LPC
//! synthesis filter ACELP subframes use, so a superframe can switch between the
//! two without a seam. Three lengths exist (20/40/80 ms, i.e. one, two or four
//! subframes' worth of samples) because a longer transform pays for itself on
//! stationary passages the way ACELP's fixed 64-sample granularity cannot.
//!
//! This module ports the pieces of `ixheaacd_tcx_mdct` that are pure signal
//! processing, each independently testable against the reference's exact
//! formulas:
//!
//! * [`weight_lpc`] — perceptual bandwidth expansion of the LPC filter
//!   (`ixheaacd_lpc_coeff_wt_apply`), the same 0.92-power weighting the FAC
//!   synthesis in [`super::acelp`] uses, reused here via
//!   [`crate::tables::usac_acelp::GAMMA_TABLE`].
//! * [`spectral_envelope_gains`] — the perceptual weighting filter's magnitude
//!   response (`ixheaacd_lpc_to_td`), used to shape white noise into the
//!   filter's own spectral envelope before shaping the coded excitation with it.
//! * [`noise_shape`] — cross-fades between the envelope at the start and the
//!   end of the frame (`ixheaacd_noise_shaping`), since a TCX frame spans
//!   several subframes and the filter is only exactly known at their boundaries.
//! * [`low_frequency_deemphasis`] — limits how much energy the lowest quarter
//!   of the spectrum can carry relative to the loudest block
//!   (`ixheaacd_low_fq_deemphasis`), which is what keeps the noise floor and
//!   FAC contribution (see below) from dominating a quiet passage.
//!
//! # What this module does not cover yet
//!
//! The pieces above shape a spectrum; turning that spectrum into a waveform and
//! splicing it against ACELP's excitation history is `ixheaacd_tcx_mdct`'s other
//! half, and is not ported: the asymmetric-overlap MDCT `ixheaacd_acelp_mdct_main`
//! (distinct from a plain square MDCT — TCX's overlap regions differ in length
//! from its transform length, exactly at the boundary these functions exist to
//! shape), the sine-window overlap-add across an ACELP/TCX/FD mode switch, and
//! the state (`mode_prev`, `exc_prev`) that carries across frame boundaries.
//! [`crate::decoder::usac::UsacDecoder::decode_lpd_frame`] still refuses any
//! `lpd_mode` other than all-ACELP, so nothing in this module is reachable from
//! that path yet — see `text/plan.txt` phase 1.7 for what remains.

use crate::dsp::fft::{Complex32, FftContext};
use crate::tables::usac_acelp::{GAMMA_TABLE, ORDER};

/// Apply the reference's perceptual bandwidth expansion to an LPC filter
/// (`ixheaacd_lpc_coeff_wt_apply`): flattening formants by shrinking each
/// coefficient's influence in proportion to its order, so the resulting filter
/// is a coarser, wider-tolerance version of the original -- exactly what a
/// perceptual weighting filter needs to be.
pub fn weight_lpc(lpc: &[f32; ORDER + 1]) -> [f32; ORDER + 1] {
    let mut out = [0.0f32; ORDER + 1];
    out[0] = lpc[0];
    for i in 1..=ORDER {
        out[i] = lpc[i] * GAMMA_TABLE[i];
    }
    out
}

/// The perceptual weighting filter's magnitude response, inverted, as a gain
/// envelope over `quarter_len` frequency bins (`ixheaacd_lpc_to_td`).
///
/// `weighted_lpc` should already have [`weight_lpc`] applied. The filter's
/// frequency response is evaluated by zero-padding its coefficients to
/// `2 * quarter_len` and taking the FFT — the direct, textbook way to sample a
/// filter's response at evenly spaced frequencies — and each gain is the
/// reciprocal of the response's magnitude there, so multiplying a flat
/// (post-noise-shaping) spectrum by these gains reintroduces the formant
/// structure `weight_lpc` flattened.
///
/// `quarter_len` must be a power of two (a quarter of the subframe count times
/// [`crate::tables::usac_acelp::LEN_SUBFR`], which is 64 for every sampling
/// rate this port supports).
pub fn spectral_envelope_gains(weighted_lpc: &[f32; ORDER + 1], quarter_len: usize) -> Vec<f32> {
    let n = 2 * quarter_len;
    debug_assert!(n.is_power_of_two() && n > ORDER);
    let mut spectrum = vec![Complex32::default(); n];
    let scale = std::f64::consts::PI / n as f64;
    for (i, coeff) in weighted_lpc.iter().enumerate() {
        let angle = i as f64 * scale;
        spectrum[i] = Complex32::new(*coeff * angle.cos() as f32, -*coeff * angle.sin() as f32);
    }
    FftContext::new(n).forward(&mut spectrum);
    spectrum[..quarter_len]
        .iter()
        .map(|c| 1.0 / ((c.re * c.re + c.im * c.im).sqrt() + f32::EPSILON))
        .collect()
}

/// Cross-fade a spectrum between two envelope-gain snapshots taken at the
/// frame's two LPC boundaries (`ixheaacd_noise_shaping`).
///
/// A TCX frame's spectral shape is only known exactly at its start and end
/// (where an LSF set was actually transmitted); everything in between is
/// interpolated. This applies that interpolation as a first-order recursive
/// filter re-parameterised every `len / bins.len()` samples — a cheap way to
/// get a smooth transition without recomputing a filter response at every
/// sample — and simultaneously reorders the result into the folded layout an
/// inverse MDCT-IV expects (even-indexed samples first in original order,
/// odd-indexed samples second in reverse order).
///
/// `gains_start` and `gains_end` must each have `spectrum.len() / block` bins,
/// where `block = spectrum.len() / gains_start.len()`.
pub fn noise_shape(spectrum: &mut [f32], gains_start: &[f32], gains_end: &[f32]) {
    let len = spectrum.len();
    let bins = gains_start.len().min(gains_end.len());
    if bins == 0 || len == 0 {
        return;
    }
    let block = len / bins;
    let mut prev = 0.0f32;
    let mut a = 0.0f32;
    let mut b = 0.0f32;
    let mut shaped = spectrum.to_vec();
    for (i, x) in shaped.iter_mut().enumerate() {
        if i % block == 0 {
            let bin = (i / block).min(bins - 1);
            let (g1, g2) = (gains_start[bin], gains_end[bin]);
            let sum = g1 + g2;
            a = if sum != 0.0 { 2.0 * g1 * g2 / sum } else { 0.0 };
            b = if sum != 0.0 { (g2 - g1) / sum } else { 0.0 };
        }
        *x = a * *x + b * prev;
        prev = *x;
    }
    let half = len / 2;
    for i in 0..half {
        spectrum[i] = shaped[2 * i];
        spectrum[half + i] = shaped[len - 2 * i - 1];
    }
}

/// Limit how much energy the lowest quarter of a TCX spectrum can carry
/// relative to its loudest 8-bin block, returning the per-block scale factors
/// applied (`ixheaacd_low_fq_deemphasis`).
///
/// TCX's lowest bins carry the FAC contribution at a mode switch (see this
/// module's top-level docs), and an unconstrained low end can dominate a quiet
/// passage; this keeps every block within a bounded ratio of the loudest one
/// in the same quarter, with a floor so a fully silent block does not get
/// scaled by an arbitrarily large factor when it is later divided out.
pub fn low_frequency_deemphasis(spectrum: &mut [f32]) -> Vec<f32> {
    const BLOCK: usize = 8;
    let span = spectrum.len() / 4;
    let mut max_energy = 0.01f32;
    for block in spectrum[..span].chunks(BLOCK) {
        let energy: f32 = 0.01 + block.iter().map(|x| x * x).sum::<f32>();
        max_energy = max_energy.max(energy);
    }
    let mut gains = Vec::with_capacity(span.div_ceil(BLOCK));
    let mut factor = 0.1f32;
    for block in spectrum[..span].chunks_mut(BLOCK) {
        let energy: f32 = 0.01 + block.iter().map(|x| x * x).sum::<f32>();
        let ratio = (energy / max_energy).sqrt();
        factor = factor.max(ratio);
        for x in block.iter_mut() {
            *x *= factor;
        }
        gains.push(factor);
    }
    gains
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighting_leaves_the_gain_term_untouched_and_shrinks_the_rest() {
        let mut lpc = [1.0f32; ORDER + 1];
        lpc[0] = 1.0;
        let weighted = weight_lpc(&lpc);
        assert_eq!(weighted[0], 1.0);
        for i in 1..=ORDER {
            assert!((weighted[i] - 0.92f32.powi(i as i32)).abs() < 1e-6);
            assert!(weighted[i].abs() < lpc[i].abs());
        }
    }

    /// A flat (all-pass) filter must produce a flat gain envelope: there is no
    /// formant to reintroduce, so every bin should come back the same.
    #[test]
    fn a_flat_filter_yields_a_flat_gain_envelope() {
        let mut lpc = [0.0f32; ORDER + 1];
        lpc[0] = 1.0;
        let gains = spectral_envelope_gains(&lpc, 32);
        assert_eq!(gains.len(), 32);
        for g in &gains {
            assert!((g - 1.0).abs() < 1e-3, "expected ~1.0, got {g}");
        }
    }

    /// A filter with an actual pole must attenuate near that pole's frequency
    /// less than it does far from it -- i.e. the envelope is not flat, and its
    /// peak sits away from bin 0 for a filter with a resonance, not at DC.
    #[test]
    fn a_resonant_filter_yields_a_shaped_envelope() {
        let mut lpc = [0.0f32; ORDER + 1];
        lpc[0] = 1.0;
        lpc[1] = -1.6;
        lpc[2] = 0.95;
        let gains = spectral_envelope_gains(&lpc, 64);
        let peak = gains.iter().cloned().fold(f32::MIN, f32::max);
        let trough = gains.iter().cloned().fold(f32::MAX, f32::min);
        assert!(peak > trough * 2.0, "expected a real resonance peak: {peak} vs {trough}");
    }

    /// With identical start/end gains the filter degenerates to `a=1, b=0`
    /// (pure pass-through) at every block, so shaping must be the identity up
    /// to the fold-and-reorder step; check that the reorder alone is exactly
    /// what the reference's index arithmetic says.
    #[test]
    fn identical_envelopes_only_reorder_the_spectrum() {
        let mut spectrum: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let gains = vec![1.0f32; 4];
        noise_shape(&mut spectrum, &gains, &gains);
        let expected: Vec<f32> = (0..8).map(|i| (2 * i) as f32).chain((0..8).map(|i| (15 - 2 * i) as f32)).collect();
        assert_eq!(spectrum, expected);
    }

    #[test]
    fn deemphasis_never_amplifies_and_keeps_the_loudest_block_near_unity() {
        let mut spectrum = vec![0.0f32; 64];
        for (i, x) in spectrum.iter_mut().enumerate() {
            *x = if i < 8 { 0.01 } else { 1.0 };
        }
        let original = spectrum.clone();
        let gains = low_frequency_deemphasis(&mut spectrum);
        assert!(gains.iter().all(|g| (0.0..=1.0 + 1e-4).contains(g)), "{gains:?}");
        // The loudest block (bins 8..16, well within the first quarter of 64)
        // must end up scaled by very close to 1.
        assert!((gains[1] - 1.0).abs() < 0.05, "{gains:?}");
        for (a, b) in spectrum[16..].iter().zip(original[16..].iter()) {
            assert_eq!(a, b, "outside the first quarter must be untouched");
        }
    }

    #[test]
    fn deemphasis_is_bounded_even_for_a_fully_silent_span() {
        let mut spectrum = vec![0.0f32; 32];
        let gains = low_frequency_deemphasis(&mut spectrum);
        assert!(gains.iter().all(|g| g.is_finite()));
        assert!(spectrum.iter().all(|x| *x == 0.0));
    }
}
