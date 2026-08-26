//! Inverse quantization of AAC spectral coefficients.
//!
//! Each coefficient is reconstructed as
//!
//! ```text
//! x = sign(q) * |q|^(4/3) * 2^((sf - 100) / 4)
//! ```
//!
//! per ISO/IEC 14496-3 clause 4.6.2. The `|q|^(4/3)` term comes from a table covering
//! the full range a conformant stream can produce, and the scalefactor term from a
//! table over the 8-bit scalefactor range, so the hot loop is two loads and a
//! multiply per coefficient.

use crate::decoder::aac::ics::{ChannelData, SF_OFFSET};
use std::sync::OnceLock;

/// Largest magnitude a conformant AAC stream may quantize to, plus the headroom
/// `pulse_data` can add on top (four pulses of amplitude 15).
pub const MAX_QUANT: usize = 8191 + 60;

/// Scalefactors are 8-bit, so `2^((sf - 100) / 4)` has 256 distinct values.
const NUM_SCALEFACTORS: usize = 256;

struct DequantTables {
    /// `|q|^(4/3)` for `q` in `0..=MAX_QUANT`.
    pow43: Vec<f32>,
    /// `2^((sf - 100) / 4)` for `sf` in `0..256`.
    gain: Vec<f32>,
}

fn tables() -> &'static DequantTables {
    static TABLES: OnceLock<DequantTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        // Built in f64 and rounded once, so the table matches a direct powf() to
        // the last f32 bit rather than accumulating error.
        let pow43 = (0..=MAX_QUANT)
            .map(|q| (q as f64).powf(4.0 / 3.0) as f32)
            .collect();
        let gain = (0..NUM_SCALEFACTORS)
            .map(|sf| ((sf as f64 - SF_OFFSET as f64) * 0.25).exp2() as f32)
            .collect();
        DequantTables { pow43, gain }
    })
}

/// `|q|^(4/3)`, falling back to `powf` beyond the table.
#[inline(always)]
fn pow43(t: &DequantTables, q_abs: usize) -> f32 {
    match t.pow43.get(q_abs) {
        Some(&v) => v,
        None => (q_abs as f32).powf(4.0 / 3.0),
    }
}

/// `2^((sf - 100) / 4)`, clamping the scalefactor to its legal 8-bit range.
#[inline(always)]
fn scale_gain(t: &DequantTables, sf: i32) -> f32 {
    t.gain[sf.clamp(0, NUM_SCALEFACTORS as i32 - 1) as usize]
}

/// Inverse-quantize a run of coefficients that share one scalefactor.
#[inline]
pub fn inverse_quantize_band(quant: &[i32], scalefactor: i32, output: &mut [f32]) {
    let t = tables();
    let gain = scale_gain(t, scalefactor);
    for (out, &q) in output.iter_mut().zip(quant.iter()) {
        let mag = pow43(t, q.unsigned_abs() as usize) * gain;
        *out = if q < 0 { -mag } else { mag };
    }
}

/// Inverse-quantize a whole channel, band by band, into `ChannelData::spec`.
///
/// Operates on the grouped layout, so band extents follow the window grouping.
/// Noise-substituted and intensity bands are left at zero here; they are filled in
/// by the PNS and intensity-stereo stages, which need the neighbouring channel.
pub fn inverse_quantize_channel(ch: &mut ChannelData) {
    let t = tables();
    ch.spec.fill(0.0);

    let ics = &ch.ics;
    for g in 0..ics.num_window_groups {
        let group_base = ics.group_base(g);
        let group_len = ics.window_group_length[g];
        for sfb in 0..ics.max_sfb {
            let cb = ch.sfb_cb[g][sfb];
            // Only Huffman-coded bands hold quantized spectrum.
            if cb == 0 || cb >= 13 {
                continue;
            }
            let start = group_base + ics.grouped_offset(g, sfb);
            let width = (ics.swb_offset[sfb + 1] - ics.swb_offset[sfb]) as usize * group_len;
            let end = (start + width).min(ch.spec.len());
            if start >= end {
                continue;
            }

            let gain = scale_gain(t, ch.scale_factors[g][sfb] as i32);
            for i in start..end {
                let q = ch.quant[i];
                let mag = pow43(t, q.unsigned_abs() as usize) * gain;
                ch.spec[i] = if q < 0 { -mag } else { mag };
            }
        }
    }
}

/// Legacy entry point kept for the encoder's round-trip checks.
pub fn inverse_quantize(quantized: &[i32], scalefactor: i16, output: &mut [f32]) {
    inverse_quantize_band(quantized, scalefactor as i32, output);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scalefactor of 100 is unity gain, so a quantized 1 must dequantize to 1.
    #[test]
    fn unity_scalefactor_is_identity_at_one() {
        let mut out = [0.0f32; 2];
        inverse_quantize_band(&[1, -1], 100, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] + 1.0).abs() < 1e-6);
    }

    /// Zero stays zero regardless of scalefactor.
    #[test]
    fn zero_stays_zero() {
        for sf in [0, 100, 255] {
            let mut out = [1.0f32; 4];
            inverse_quantize_band(&[0; 4], sf, &mut out);
            assert_eq!(out, [0.0; 4]);
        }
    }

    /// The table must agree with the defining power law across its whole range.
    #[test]
    fn pow43_table_matches_the_power_law() {
        let t = tables();
        for q in [0usize, 1, 2, 3, 15, 16, 100, 1023, 1024, 4095, 8191, MAX_QUANT] {
            let expect = (q as f64).powf(4.0 / 3.0);
            let got = pow43(t, q) as f64;
            let tol = expect.abs() * 1e-6 + 1e-6;
            assert!((got - expect).abs() <= tol, "q={q}: {got} vs {expect}");
        }
    }

    /// Each step of four in the scalefactor must double the gain.
    #[test]
    fn scalefactor_gain_doubles_every_four_steps() {
        let t = tables();
        for sf in 4..252 {
            let lo = scale_gain(t, sf) as f64;
            let hi = scale_gain(t, sf + 4) as f64;
            assert!((hi / lo - 2.0).abs() < 1e-5, "sf {sf}: ratio {}", hi / lo);
        }
    }

    /// Magnitudes past the table must still follow the power law.
    #[test]
    fn out_of_table_magnitudes_fall_back() {
        let t = tables();
        let q = MAX_QUANT + 1000;
        let expect = (q as f32).powf(4.0 / 3.0);
        assert!((pow43(t, q) - expect).abs() <= expect * 1e-5);
    }

    /// Sign must follow the quantized value, not the scalefactor.
    #[test]
    fn sign_is_preserved() {
        let mut out = [0.0f32; 4];
        inverse_quantize_band(&[5, -5, 200, -200], 120, &mut out);
        assert!(out[0] > 0.0 && out[1] < 0.0 && out[2] > 0.0 && out[3] < 0.0);
        assert!((out[0] + out[1]).abs() < 1e-3);
        assert!((out[2] + out[3]).abs() < 1e-1);
    }
}
