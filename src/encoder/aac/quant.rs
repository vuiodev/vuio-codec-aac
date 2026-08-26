//! Quantization and rate control.
//!
//! AAC quantizes a spectral coefficient as
//!
//! ```text
//! q = sign(x) * floor(|x| ^ (3/4) * 2^(-3/16 * (sf - 100)) + 0.4054)
//! ```
//!
//! which the decoder inverts with `|q|^(4/3) * 2^((sf - 100)/4)`. The scalefactor
//! `sf` sets the step size for a band: raising it by four halves the resolution and
//! roughly halves the bits that band costs.
//!
//! Rate control here is a bisection on a single global scalefactor. That produces a
//! correct, decodable stream at close to the requested bitrate, with a flat noise
//! floor rather than a perceptually shaped one; see [`crate::encoder::aac::psycho`]
//! for the masking model that would shape it per band.

use crate::bitstream::BitWriter;
use crate::encoder::aac::huffman::{minimum_codebook, tuple_cost, write_tuple};
use crate::decoder::aac::huffman::codebook;

/// Scalefactor that corresponds to unity step size.
pub const SF_OFFSET: i32 = 100;

/// Rounding bias the standard's quantizer uses.
const QUANT_BIAS: f32 = 0.4054;

/// Largest magnitude a conformant stream may quantize to.
pub const MAX_QUANT_MAGNITUDE: i32 = 8191;

/// Quantize one band with the given scalefactor.
#[inline]
pub fn quantize_band(spectrum: &[f32], scalefactor: i32, quantized: &mut [i32]) {
    debug_assert_eq!(spectrum.len(), quantized.len());
    // 2^(-3/16 * (sf - 100)) folds the scalefactor into the 3/4-power law.
    let step = (-0.1875 * (scalefactor - SF_OFFSET) as f32).exp2();

    for (q, &x) in quantized.iter_mut().zip(spectrum.iter()) {
        let mag = x.abs();
        if mag <= 0.0 {
            *q = 0;
            continue;
        }
        // x^(3/4) via two square roots, which is exact enough here and much
        // cheaper than a general power.
        let p34 = (mag * mag.sqrt()).sqrt();
        let v = (p34 * step + QUANT_BIAS) as i32;
        let v = v.min(MAX_QUANT_MAGNITUDE);
        *q = if x < 0.0 { -v } else { v };
    }
}

/// Per-band coding decision.
#[derive(Debug, Clone, Copy, Default)]
pub struct BandChoice {
    /// Chosen spectral codebook; 0 means the band is not coded.
    pub codebook: u8,
    /// Bits the band's spectral data costs.
    pub bits: u32,
}

/// Pick the cheapest codebook for a band and report its cost.
///
/// AAC pairs each codebook with an alternative tuned for a different distribution,
/// so both candidates that can represent the band are measured and the cheaper wins.
pub fn choose_codebook(band: &[i32]) -> BandChoice {
    let Some(base) = minimum_codebook(band) else {
        return BandChoice { codebook: 0, bits: 0 };
    };

    let mut best = BandChoice { codebook: 0, bits: u32::MAX };
    // Each codebook has a companion one number higher tuned for the same range.
    for cb in [base, base + 1] {
        let Some(book) = codebook(cb) else { continue };
        if band.len() % book.dim != 0 {
            continue;
        }
        let mut total = 0u32;
        let mut ok = true;
        for chunk in band.chunks_exact(book.dim) {
            match tuple_cost(cb, chunk) {
                Some(c) => total += c,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && total < best.bits {
            best = BandChoice { codebook: cb, bits: total };
        }
    }

    if best.bits == u32::MAX { BandChoice { codebook: 0, bits: 0 } } else { best }
}

/// Write a band's spectral data with the chosen codebook.
pub fn write_band(writer: &mut BitWriter, cb: u8, band: &[i32]) -> bool {
    if cb == 0 {
        return true;
    }
    let Some(book) = codebook(cb) else { return false };
    for chunk in band.chunks_exact(book.dim) {
        if !write_tuple(writer, cb, chunk) {
            return false;
        }
    }
    true
}

/// Estimate a starting scalefactor from the band's peak magnitude.
///
/// Picks the scalefactor that would put the peak near the top of the target
/// codebook range, which lands bisection close to its answer on the first try.
pub fn initial_scalefactor(spectrum: &[f32], target_peak: f32) -> i32 {
    let peak = spectrum.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if peak <= 0.0 || target_peak <= 0.0 {
        return SF_OFFSET;
    }
    // Solve peak^(3/4) * 2^(-3/16 * (sf - 100)) = target_peak for sf.
    let p34 = (peak * peak.sqrt()).sqrt();
    let sf = SF_OFFSET as f32 + (p34 / target_peak).log2() / 0.1875;
    (sf.round() as i32).clamp(0, 255)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::aac::dequant::inverse_quantize_band;

    /// Round-trip error must stay inside one quantizer step.
    ///
    /// A scalar quantizer's reconstruction error is bounded by the spacing of the
    /// reconstruction levels around the value, not by any fixed relative figure: at
    /// a coarse scalefactor a small coefficient legitimately lands far from its
    /// input in relative terms because the neighbouring levels are far apart.
    #[test]
    fn quantize_dequantize_stays_within_one_step() {
        for sf in [80i32, 100, 120, 140] {
            let spectrum: Vec<f32> =
                (1..200).map(|i| (i as f32) * 13.7 * if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
            let mut q = vec![0i32; spectrum.len()];
            quantize_band(&spectrum, sf, &mut q);
            let mut back = vec![0.0f32; spectrum.len()];
            inverse_quantize_band(&q, sf, &mut back);

            for (i, (&x, &y)) in spectrum.iter().zip(back.iter()).enumerate() {
                // A coefficient that survives quantization must keep its sign;
                // one that quantizes to zero has no sign to keep.
                if y != 0.0 {
                    assert_eq!(
                        x.is_sign_negative(),
                        y.is_sign_negative(),
                        "sign flipped at {i}: {x} -> {y}"
                    );
                }

                // Spacing between the reconstruction levels bracketing |q|.
                let mag = q[i].unsigned_abs() as i32;
                let mut levels = [0.0f32; 2];
                inverse_quantize_band(&[mag, mag + 1], sf, &mut levels);
                let step = (levels[1] - levels[0]).abs().max(1e-6);

                assert!(
                    (x.abs() - y.abs()).abs() <= step,
                    "sf {sf} index {i}: {x} -> {y}, off by more than one step of {step}"
                );
            }
        }
    }

    /// Quantization must not invent energy: a reconstructed magnitude may not exceed
    /// the input by more than one step.
    #[test]
    fn quantization_does_not_amplify() {
        let spectrum: Vec<f32> = (1..500).map(|i| i as f32 * 7.3).collect();
        for sf in [90i32, 110, 130] {
            let mut q = vec![0i32; spectrum.len()];
            quantize_band(&spectrum, sf, &mut q);
            let mut back = vec![0.0f32; spectrum.len()];
            inverse_quantize_band(&q, sf, &mut back);
            let in_energy: f64 = spectrum.iter().map(|v| (*v as f64).powi(2)).sum();
            let out_energy: f64 = back.iter().map(|v| (*v as f64).powi(2)).sum();
            assert!(
                out_energy <= in_energy * 1.5,
                "sf {sf}: energy grew from {in_energy} to {out_energy}"
            );
        }
    }

    /// A higher scalefactor must never produce larger quantized magnitudes.
    #[test]
    fn larger_scalefactor_quantizes_coarser() {
        let spectrum: Vec<f32> = (1..64).map(|i| i as f32 * 100.0).collect();
        let mut prev: Option<Vec<i32>> = None;
        for sf in (100..=160).step_by(4) {
            let mut q = vec![0i32; spectrum.len()];
            quantize_band(&spectrum, sf, &mut q);
            if let Some(p) = &prev {
                for (a, b) in p.iter().zip(q.iter()) {
                    assert!(b.abs() <= a.abs(), "sf {sf}: {b} exceeds {a}");
                }
            }
            prev = Some(q);
        }
    }

    /// Silence must quantize to zero and cost nothing.
    #[test]
    fn silence_costs_nothing() {
        let mut q = vec![0i32; 16];
        quantize_band(&vec![0.0; 16], 100, &mut q);
        assert!(q.iter().all(|&v| v == 0));
        let choice = choose_codebook(&q);
        assert_eq!(choice.codebook, 0);
        assert_eq!(choice.bits, 0);
    }

    /// The chosen codebook must be able to write the band, and the reported cost
    /// must equal the bits actually written.
    #[test]
    fn chosen_codebook_cost_is_exact() {
        for peak in [1i32, 2, 4, 7, 12, 40, 500, 8000] {
            let band: Vec<i32> = (0..16)
                .map(|i| if i % 4 == 0 { peak } else { (i as i32 % 3) - 1 })
                .collect();
            let choice = choose_codebook(&band);
            assert_ne!(choice.codebook, 0, "peak {peak} produced no codebook");

            let mut w = BitWriter::with_capacity(64);
            let before = w.bits_written();
            assert!(write_band(&mut w, choice.codebook, &band), "peak {peak}");
            assert_eq!((w.bits_written() - before) as u32, choice.bits, "peak {peak}");
        }
    }

    /// The initial estimate must put the peak near the requested magnitude.
    #[test]
    fn initial_scalefactor_lands_near_target() {
        for peak in [10.0f32, 1000.0, 30000.0] {
            let spectrum = vec![peak; 32];
            let sf = initial_scalefactor(&spectrum, 8.0);
            let mut q = vec![0i32; 32];
            quantize_band(&spectrum, sf, &mut q);
            let got = q[0].unsigned_abs() as f32;
            assert!(
                (2.0..=32.0).contains(&got),
                "peak {peak}: sf {sf} quantized to {got}, expected near 8"
            );
        }
    }
}
