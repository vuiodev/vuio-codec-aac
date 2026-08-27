//! Temporal Noise Shaping for the USAC FD path.
//!
//! Ported from `iusace_tns_usac.c`'s long-window branch, the only one this minimal
//! codec ([`crate::encoder::usac::fd`]) needs: fixed 1024-sample long windows, no
//! short-window TNS. Where the algorithm coincides with classic AAC-LC's TNS (the
//! reflection-coefficient-to-LPC step-up, the arcsine-quantized coefficient table at
//! 4-bit resolution, and the forward/inverse filter recursion itself), this reuses
//! [`crate::decoder::aac::tns`] directly rather than re-deriving it — confirmed
//! against the C source, not assumed:
//!
//! * `iusace_step_up` is byte-for-byte the same recursion as
//!   [`parcor_to_lpc`](crate::decoder::aac::tns::parcor_to_lpc) (`a[i] += k*a[order-i]`).
//! * `iusace_quantize_reflection_coeffs` at `coeff_res = 4` (the long-window value)
//!   uses the exact same `iqfac`/`iqfac_m` arcsine formula that built
//!   [`TNS_PARCOR_4`](crate::decoder::aac::tns::TNS_PARCOR_4), and the reference
//!   decoder's `ixheaacd_tns_dec_coef_usac` indexes it with the same `+8` bias
//!   `filter_parcor` uses for AAC-LC's 4-bit table.
//! * `iusace_tns_filter`'s `direction == 0` case — the only direction USAC's FD
//!   encoder ever emits for a long window (`tns_filter->direction = 0;` is
//!   hardcoded in `iusace_tns_encode`) — stores each *original* sample before
//!   overwriting it and feeds those originals back into later samples with `+=`:
//!   that is exactly [`ma_filter`](crate::decoder::aac::tns::ma_filter)'s all-zero,
//!   feed-forward form with `descending = false`. The reference decoder's inverse
//!   (`ixheaacd_tns_ar_filter_usac`) is the matching all-pole feedback form, exactly
//!   [`ar_filter`](crate::decoder::aac::tns::ar_filter).
//!
//! What is genuinely USAC-specific and ported fresh here: the frequency-to-band
//! mapping, the perceptually-weighted (whitened) spectrum TNS analyses instead of
//! the raw one, and the Gaussian lag window — none of these have an AAC-LC
//! equivalent in this codebase to reuse.

use crate::decoder::aac::ics::MAX_TNS_ORDER;
use crate::decoder::aac::tns::{TNS_PARCOR_4, ma_filter, parcor_to_lpc};

/// Sample rate this fixed configuration is tuned for, matching
/// [`crate::encoder::usac::fd`]'s `MODEL_SAMPLE_RATE_HZ`.
const SAMPLE_RATE_HZ: f64 = 44_100.0;
/// Low edge of the LPC analysis window, in Hz (`lpc_start_freq_long` in
/// `iusace_tns_init`).
const LPC_START_FREQ_HZ: f64 = 2500.0;
/// High edge of the LPC analysis window, in Hz (`lpc_stop_freq`).
const LPC_STOP_FREQ_HZ: f64 = 16000.0;
/// Lowest scalefactor band TNS may shape, for a long window at 44.1 kHz
/// (`iusace_tns_min_band_number_long[4]`, index 4 being the reference's 44100 Hz
/// row).
pub const MIN_START_BAND: usize = 17;
/// Highest scalefactor band TNS may reach, for a long window at 44.1 kHz
/// (`iusace_tns_max_bands_table[4][0]`).
pub const MAX_STOP_BAND: usize = 42;
/// Filter order the analysis searches (`tns_max_order_long`).
pub const ORDER: usize = 15;
/// Reflection-coefficient bit width for long windows (the `coeff_res = 4` set in
/// `iusace_tns_encode`'s long-window branch).
pub const COEF_RES_BITS: u32 = 4;
/// Prediction gain below which the filter costs more in side info than the
/// residual it saves (`DEF_TNS_GAIN_THRESH`).
const GAIN_THRESHOLD: f64 = 1.41;
/// Reflection coefficients this small are truncated rather than transmitted
/// (`DEF_TNS_COEFF_THRESH`).
const COEFF_THRESHOLD: f32 = 0.1;
/// Analysis window's time-resolution parameter, in the units `iusace_calc_gauss_win`
/// expects. `iusace_tns_init` sets this to `0.5` once the bitrate clears 36 kbit/s
/// per channel, which is the bracket [`crate::encoder::usac::fd`]'s fixed
/// `MODEL_BITRATE_BPS` (64 kbit/s) falls into; this encoder does not thread a real
/// per-frame bitrate through, so the value is fixed rather than branched on.
const TIME_RESOLUTION: f64 = 0.5;

/// A TNS decision for one frame: the quantized reflection-coefficient indices the
/// bitstream carries, and how many of them are in use.
#[derive(Debug, Clone, Copy, Default)]
pub struct TnsFilter {
    pub order: usize,
    pub coef: [i8; MAX_TNS_ORDER],
}

/// Precomputed, frame-independent setup: the band range analysis measures over, the
/// band range the filter is applied to, and the Gaussian lag window. Built once and
/// reused every frame, the same way [`crate::encoder::usac::fd::Layout`] is.
pub struct TnsSetup {
    lpc_start_band: usize,
    lpc_stop_band: usize,
    start_band: usize,
    stop_band: usize,
    window: [f64; ORDER + 1],
}

/// The band range the filter is actually applied to (as opposed to the range
/// analysis measures over) — a pure function of `num_sfb`, so the decoder can
/// recompute exactly the same range the encoder used without either side
/// transmitting it (matching the reference decoder, which also derives this
/// from its own copy of the sample-rate/window-type table rather than reading
/// it from the bitstream).
pub fn filter_band_range(num_sfb: usize) -> (usize, usize) {
    (MIN_START_BAND.min(num_sfb), MAX_STOP_BAND.min(num_sfb))
}

impl TnsSetup {
    pub fn new(sfb_offsets: &[usize], num_sfb: usize) -> Self {
        let lpc_start_band = freq_to_band(LPC_START_FREQ_HZ, num_sfb, sfb_offsets);
        let lpc_stop_band = freq_to_band(LPC_STOP_FREQ_HZ, num_sfb, sfb_offsets);
        let (start_band, stop_band) = filter_band_range(num_sfb);
        Self { lpc_start_band, lpc_stop_band, start_band, stop_band, window: gaussian_window() }
    }

    /// The `length` field `write_tns_data` transmits: purely informational (the
    /// reference decoder recomputes `start_band`/`stop_band` itself from the same
    /// fixed sample-rate/window-type table, the same way this decoder does), but
    /// carried for bitstream shape fidelity.
    pub fn length_field(&self, num_sfb: usize) -> u8 {
        (num_sfb.saturating_sub(self.start_band)).min(63) as u8
    }
}

/// `iusace_freq_to_band_mapping`: which scalefactor band a frequency falls
/// closest to, given the band table's total line count and the fixed sample rate.
/// Kept in the reference's integer arithmetic throughout, since the two sides only
/// have to agree with each other, not with a floating-point reproduction of it.
fn freq_to_band(freq_hz: f64, num_bands: usize, offsets: &[usize]) -> usize {
    let total = offsets[num_bands] as i64;
    let sample_rate = SAMPLE_RATE_HZ as i64;
    let line_num = ((freq_hz as i64) * total * 4 / sample_rate + 1) / 2;

    if line_num >= total {
        return num_bands;
    }

    let mut band = 0usize;
    while band < num_bands && offsets[band + 1] as i64 <= line_num {
        band += 1;
    }
    if band < num_bands
        && (line_num - offsets[band] as i64) > (offsets[band + 1] as i64 - line_num)
    {
        band += 1;
    }
    band
}

/// `iusace_calc_gauss_win`, at this module's fixed sample rate, order and time
/// resolution.
fn gaussian_window() -> [f64; ORDER + 1] {
    let gauss_exp = std::f64::consts::PI * SAMPLE_RATE_HZ * 0.001 * TIME_RESOLUTION / 1024.0;
    let gauss_exp = -0.5 * gauss_exp * gauss_exp;
    let mut window = [0.0f64; ORDER + 1];
    for (i, w) in window.iter_mut().enumerate() {
        let x = i as f64 + 0.5;
        *w = (gauss_exp * x * x).exp();
    }
    window
}

/// Build the perceptually-whitened spectrum TNS analyses, over
/// `[setup.lpc_start_band, setup.lpc_stop_band)`: each line is divided by the
/// square root of its own band's energy (so a loud band does not dominate the
/// autocorrelation the way it would in the raw spectrum), then smoothed against its
/// neighbours — `iusace_calc_weighted_spec`.
fn weighted_spectrum(spectrum: &[f32], sfb_offsets: &[usize], setup: &TnsSetup) -> Vec<f64> {
    let lo = sfb_offsets[setup.lpc_start_band];
    let hi = sfb_offsets[setup.lpc_stop_band].min(spectrum.len());

    let mut weight = vec![0.0f64; hi];
    let mut band = setup.lpc_start_band;
    let band_weight = |b: usize| -> f64 {
        let blo = sfb_offsets[b];
        let bhi = sfb_offsets[b + 1].min(spectrum.len());
        let energy: f64 = spectrum[blo..bhi].iter().map(|&x| (x as f64) * (x as f64)).sum();
        1.0 / (energy + 1e-30).sqrt()
    };
    let mut current = band_weight(band);

    for (i, w) in weight.iter_mut().enumerate().take(hi).skip(lo) {
        if sfb_offsets[band + 1] == i {
            band += 1;
            if band + 1 < setup.lpc_stop_band {
                current = band_weight(band);
            }
        }
        *w = current;
    }

    for i in (lo..hi.saturating_sub(1)).rev() {
        weight[i] = (weight[i] + weight[i + 1]) * 0.5;
    }
    for i in (lo + 1)..hi {
        weight[i] = (weight[i] + weight[i - 1]) * 0.5;
    }

    (lo..hi).map(|i| weight[i] * spectrum[i] as f64).collect()
}

/// Index of the [`TNS_PARCOR_4`] entry closest to `value`.
fn nearest(value: f32) -> usize {
    let mut best = 0usize;
    let mut best_distance = f32::INFINITY;
    for (i, &entry) in TNS_PARCOR_4.iter().enumerate() {
        let distance = (entry - value).abs();
        if distance < best_distance {
            best_distance = distance;
            best = i;
        }
    }
    best
}

/// Analyse one frame's spectrum for short-term structure across frequency and, if
/// shaping it is worth the filter's bits, apply the analysis (all-zero) filter in
/// place and return the filter to transmit. Returns `None` when the spectrum is
/// flat enough over the range TNS may touch that filtering would not pay — in
/// which case `spectrum` is left untouched.
pub fn apply(spectrum: &mut [f32], sfb_offsets: &[usize], setup: &TnsSetup) -> Option<TnsFilter> {
    let lpc_lo = sfb_offsets[setup.lpc_start_band];
    let lpc_hi = sfb_offsets[setup.lpc_stop_band].min(spectrum.len());
    if lpc_hi <= lpc_lo + ORDER {
        return None;
    }

    let weighted = weighted_spectrum(spectrum, sfb_offsets, setup);

    let mut autocorrelation = [0.0f64; ORDER + 1];
    for (lag, slot) in autocorrelation.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        for i in lag..weighted.len() {
            sum += weighted[i] * weighted[i - lag];
        }
        *slot = sum;
    }
    if autocorrelation[0] <= 0.0 {
        return None;
    }
    for (lag, slot) in autocorrelation.iter_mut().enumerate() {
        *slot *= setup.window[lag];
    }

    // Levinson-Durbin, producing reflection coefficients directly (the same
    // structure as encoder::aac::tns::analyse, over this module's own order and
    // windowed autocorrelation).
    let mut lpc = [0.0f64; ORDER + 1];
    let mut work = [0.0f64; ORDER + 1];
    let mut error = autocorrelation[0];
    lpc[0] = 1.0;
    let mut parcor = [0.0f32; ORDER];
    let mut reached = 0usize;

    for m in 1..=ORDER {
        let mut acc = autocorrelation[m];
        for i in 1..m {
            acc += lpc[i] * autocorrelation[m - i];
        }
        let k = if error.abs() > f64::MIN_POSITIVE { -acc / error } else { 0.0 };
        let k = k.clamp(-0.999, 0.999);
        parcor[m - 1] = k as f32;
        reached = m;

        work[..=m].copy_from_slice(&lpc[..=m]);
        for i in 1..m {
            lpc[i] = work[i] + k * work[m - i];
        }
        lpc[m] = k;
        error *= 1.0 - k * k;
        if error <= 0.0 {
            break;
        }
    }

    let gain = if error > f64::MIN_POSITIVE { autocorrelation[0] / error } else { 1.0 };
    if gain <= GAIN_THRESHOLD {
        return None;
    }

    // Quantize every coefficient the recursion reached, then truncate from the top:
    // the highest index whose quantized magnitude clears the threshold sets the
    // transmitted order, exactly `iusace_truncate_coeffs`'s trailing-zero trim.
    let mut coef = [0i8; MAX_TNS_ORDER];
    let mut quantized = [0.0f32; ORDER];
    let bias = (TNS_PARCOR_4.len() / 2) as i32;
    let mut order = 0usize;
    for i in 0..reached {
        let idx = nearest(parcor[i]) as i32 - bias;
        coef[i] = idx as i8;
        quantized[i] = TNS_PARCOR_4[(idx + bias) as usize];
        if quantized[i].abs() > COEFF_THRESHOLD {
            order = i + 1;
        }
    }
    if order == 0 {
        return None;
    }

    let mut lpc32 = [0.0f32; MAX_TNS_ORDER + 1];
    parcor_to_lpc(&quantized[..order], &mut lpc32);

    let apply_lo = sfb_offsets[setup.start_band];
    let apply_hi = sfb_offsets[setup.stop_band].min(spectrum.len());
    if apply_hi <= apply_lo {
        return None;
    }
    ma_filter(&mut spectrum[apply_lo..apply_hi], &lpc32, order, false);

    Some(TnsFilter { order, coef })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::aac::tns::ar_filter;
    use crate::tables::scalefactor::{MAX_SFB_LONG, compute_sfb_offsets};
    use crate::tables::sfb::SFB_48_1024;

    fn layout() -> ([usize; MAX_SFB_LONG + 1], usize) {
        let mut offsets = [0usize; MAX_SFB_LONG + 1];
        let count = compute_sfb_offsets(SFB_48_1024, &mut offsets);
        (offsets, count - 1)
    }

    fn structured_spectrum(lines: usize) -> Vec<f32> {
        (0..lines)
            .map(|i| {
                let envelope = (-(i as f32) / 300.0).exp();
                1.0e5 * envelope * (i as f32 * 0.31).sin()
            })
            .collect()
    }

    /// A spectrum with real short-term structure across the range TNS analyses
    /// must both trigger the filter and leave it invertible: this is the
    /// end-to-end property the whole tool depends on.
    #[test]
    fn filtering_reduces_the_residual_and_inverts_exactly() {
        let (offsets, num_sfb) = layout();
        let setup = TnsSetup::new(&offsets[..=num_sfb], num_sfb);
        let original = structured_spectrum(1024);

        let mut spectrum = original.clone();
        let filter = apply(&mut spectrum, &offsets, &setup)
            .expect("a spectrum with this much structure must be worth shaping");

        let lo = offsets[setup.start_band];
        let hi = offsets[setup.stop_band];
        let before: f64 = original[lo..hi].iter().map(|&v| (v as f64) * v as f64).sum();
        let after: f64 = spectrum[lo..hi].iter().map(|&v| (v as f64) * v as f64).sum();
        assert!(after < before, "filtering left {after} where {before} went in");

        let mut quantized = [0.0f32; ORDER];
        let bias = (TNS_PARCOR_4.len() / 2) as i32;
        for i in 0..filter.order {
            quantized[i] = TNS_PARCOR_4[(filter.coef[i] as i32 + bias) as usize];
        }
        let mut lpc = [0.0f32; MAX_TNS_ORDER + 1];
        parcor_to_lpc(&quantized[..filter.order], &mut lpc);
        ar_filter(&mut spectrum[lo..hi], &lpc, filter.order, false);

        for (i, (&want, &got)) in original[lo..hi].iter().zip(spectrum[lo..hi].iter()).enumerate() {
            let tolerance = 1e-2 * want.abs().max(1.0);
            assert!((want - got).abs() <= tolerance, "line {i}: {got} vs {want}");
        }
    }

    /// White noise has nothing predictable across frequency, so no filter should
    /// be worth transmitting, and the spectrum must be left exactly as it was.
    #[test]
    fn unstructured_spectrum_is_left_alone() {
        let (offsets, num_sfb) = layout();
        let setup = TnsSetup::new(&offsets[..=num_sfb], num_sfb);

        let mut state = 987654u32;
        let spectrum: Vec<f32> = (0..1024)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1 << 23) as f32 - 1.0) * 1.0e4
            })
            .collect();

        let mut work = spectrum.clone();
        let filter = apply(&mut work, &offsets, &setup);
        assert!(filter.is_none(), "unstructured noise must not be worth a TNS filter");
        assert_eq!(work, spectrum, "the spectrum was touched despite no filter being used");
    }
}
