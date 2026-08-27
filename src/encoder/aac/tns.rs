//! Temporal noise shaping, encoder side.
//!
//! Quantization noise is flat across the window the transform covers. On a signal
//! whose envelope moves quickly inside that window — speech, a plucked string, any
//! attack — flat noise is heard as a smear before and after the event. Temporal
//! noise shaping fixes that by running a prediction filter *along the frequency
//! axis*: the residual is what gets quantized, so the noise the decoder's inverse
//! filter puts back follows the same envelope as the signal and hides under it.
//!
//! The encoder's job, per frame:
//!
//! 1. measure how predictable the spectrum is over the range TNS may touch,
//! 2. if predicting it is worth the filter's cost, quantize the reflection
//!    coefficients the bitstream carries,
//! 3. filter the spectrum with the all-zero form of exactly those coefficients, so
//!    that the decoder's all-pole filter is its exact inverse.
//!
//! Step 3 has to use the *quantized* coefficients, not the ones the analysis found:
//! anything else leaves the decoder undoing a filter that was never applied.

use crate::decoder::aac::ics::{MAX_TNS_ORDER, TnsFilterSpec};
use crate::decoder::aac::tns::{TNS_PARCOR_3, TNS_PARCOR_4, ma_filter, parcor_to_lpc, tns_max_band};
use crate::types::SamplingRate;

/// Prediction gain below which the filter is not worth its bits.
///
/// A filter costs about forty bits; below this the residual it saves does not pay
/// for them, and shaping noise the ear was not going to notice risks making things
/// worse rather than better.
const MIN_PREDICTION_GAIN: f32 = 1.4;
/// Order the encoder searches up to.
const ORDER: usize = 12;
/// Lag window applied to the autocorrelation, in Hz of equivalent bandwidth.
///
/// Broadening the autocorrelation slightly keeps the filter away from the unit
/// circle, where quantizing a reflection coefficient can turn a stable filter into
/// one that rings.
const LAG_WINDOW_BANDWIDTH: f32 = 0.02;
/// How far a reflection coefficient may sit from the unit circle.
const MAX_PARCOR: f32 = 0.99;
/// Lowest frequency the filter is allowed to touch, in Hz.
///
/// Below this the transform's own resolution already follows the ear, and filtering
/// across frequency there would spread noise between critical bands rather than
/// shape it in time.
const START_FREQUENCY_HZ: f32 = 1500.0;

/// A TNS decision for one window.
#[derive(Debug, Clone, Copy, Default)]
pub struct TnsFilter {
    /// The filter as the bitstream will carry it.
    pub spec: TnsFilterSpec,
    /// Prediction gain the analysis found, for reporting.
    pub gain: f32,
}

/// Decide whether to shape this window's noise, and shape it if so.
///
/// `spectrum` is one window's coefficients, filtered in place. `offsets` is the
/// band table and `bands` the highest band the channel codes. Returns the filter to
/// transmit, or `None` when the spectrum is flat enough that shaping would not pay.
///
/// The range is clipped exactly the way the decoder clips it, so the two ends of
/// the chain filter the same lines whatever the band table happens to be.
pub fn apply(
    spectrum: &mut [f32],
    offsets: &[usize],
    bands: usize,
    rate: SamplingRate,
    short: bool,
) -> Option<TnsFilter> {
    let stop_band = tns_max_band(rate, short).min(bands);
    let start_band = first_band_above(offsets, stop_band, rate, short);
    if start_band >= stop_band {
        return None;
    }
    let lo = offsets[start_band];
    let hi = offsets[stop_band].min(spectrum.len());
    if hi <= lo + ORDER {
        return None;
    }

    let (parcor, gain) = analyse(&spectrum[lo..hi]);
    if gain < MIN_PREDICTION_GAIN {
        return None;
    }

    // Quantize first, then filter with what was quantized: the decoder has nothing
    // else to invert with.
    let resolution: u8 = if short { 0 } else { 1 };
    let table: &[f32] = if resolution == 0 { &TNS_PARCOR_3 } else { &TNS_PARCOR_4 };
    let bias = (table.len() / 2) as i32;

    let mut spec = TnsFilterSpec {
        start_band,
        stop_band,
        order: 0,
        downward: false,
        coef: [0; MAX_TNS_ORDER],
        resolution,
    };
    let mut quantized = [0.0f32; MAX_TNS_ORDER];
    for (i, &k) in parcor.iter().enumerate() {
        let index = nearest(table, k) as i32 - bias;
        spec.coef[i] = index as i8;
        quantized[i] = table[(index + bias) as usize];
        if index != 0 {
            spec.order = i + 1;
        }
    }
    if spec.order == 0 {
        return None;
    }

    let mut lpc = [0.0f32; MAX_TNS_ORDER + 1];
    parcor_to_lpc(&quantized[..spec.order], &mut lpc);
    ma_filter(&mut spectrum[lo..hi], &lpc, spec.order, spec.downward);

    Some(TnsFilter { spec, gain })
}

/// First band whose lower edge reaches [`START_FREQUENCY_HZ`].
fn first_band_above(offsets: &[usize], stop_band: usize, rate: SamplingRate, short: bool) -> usize {
    // The frame's whole spectrum spans half the sampling rate, over as many lines
    // as the window is long.
    let lines = if short { 128 } else { 1024 } as f32;
    let per_line = rate.hz() as f32 * 0.5 / lines;
    for b in 0..stop_band {
        if offsets[b] as f32 * per_line >= START_FREQUENCY_HZ {
            return b;
        }
    }
    stop_band
}

/// Index of the table entry closest to `value`.
fn nearest(table: &[f32], value: f32) -> usize {
    let mut best = 0usize;
    let mut best_distance = f32::INFINITY;
    for (i, &entry) in table.iter().enumerate() {
        let distance = (entry - value).abs();
        if distance < best_distance {
            best_distance = distance;
            best = i;
        }
    }
    best
}

/// Fit a prediction filter across frequency and report how much it predicts.
///
/// Returns the reflection coefficients and the prediction gain, the ratio of the
/// spectrum's own power to what the filter leaves behind. A gain near one means the
/// spectrum is unstructured and nothing is worth transmitting.
fn analyse(spectrum: &[f32]) -> ([f32; MAX_TNS_ORDER], f32) {
    let mut autocorrelation = [0.0f64; ORDER + 1];
    for (lag, slot) in autocorrelation.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        for i in lag..spectrum.len() {
            sum += spectrum[i] as f64 * spectrum[i - lag] as f64;
        }
        *slot = sum;
    }

    let mut parcor = [0.0f32; MAX_TNS_ORDER];
    if autocorrelation[0] <= 0.0 {
        return (parcor, 1.0);
    }

    // Widening the autocorrelation a little is what keeps a nearly-periodic spectrum
    // from producing a filter that sits on the unit circle.
    for (lag, slot) in autocorrelation.iter_mut().enumerate().skip(1) {
        let x = LAG_WINDOW_BANDWIDTH * lag as f32;
        *slot *= (-0.5 * (x * x) as f64).exp();
    }

    // Levinson-Durbin, which produces the reflection coefficients directly.
    let mut lpc = [0.0f64; ORDER + 1];
    let mut work = [0.0f64; ORDER + 1];
    let mut error = autocorrelation[0];
    lpc[0] = 1.0;

    for m in 1..=ORDER {
        let mut acc = autocorrelation[m];
        for i in 1..m {
            acc += lpc[i] * autocorrelation[m - i];
        }
        let k = if error.abs() > f64::MIN_POSITIVE { -acc / error } else { 0.0 };
        let k = k.clamp(-MAX_PARCOR as f64, MAX_PARCOR as f64);
        parcor[m - 1] = k as f32;

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

    let gain = if error > f64::MIN_POSITIVE { (autocorrelation[0] / error) as f32 } else { 1.0 };
    (parcor, gain.max(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::aac::tns::ar_filter;

    /// The real 44.1 kHz long-window band table, so the frequency limits the
    /// encoder works from mean what they say.
    fn real_table() -> (Vec<usize>, usize) {
        use crate::tables::scalefactor::{MAX_SFB_LONG, compute_sfb_offsets, get_sfb_table};
        use crate::types::FrameLength;
        let widths = get_sfb_table(SamplingRate::Hz44100, false, FrameLength::Samples1024);
        let mut offsets = [0usize; MAX_SFB_LONG + 1];
        let count = compute_sfb_offsets(widths, &mut offsets);
        (offsets[..count].to_vec(), count - 1)
    }

    /// The decoder's synthesis filter must undo the encoder's analysis filter
    /// exactly, or TNS would corrupt the spectrum rather than shape its noise.
    #[test]
    fn the_decoder_filter_inverts_the_encoder_filter() {
        let (offsets, bands) = real_table();
        let lines = offsets[bands];
        // A spectrum with a strong envelope across frequency, which is what TNS is
        // there to exploit.
        let original: Vec<f32> = (0..lines)
            .map(|i| {
                let envelope = (-(i as f32) / 120.0).exp();
                1.0e5 * envelope * (i as f32 * 0.7).sin()
            })
            .collect();

        let mut spectrum = original.clone();
        let filter = apply(&mut spectrum, &offsets, bands, SamplingRate::Hz44100, false)
            .expect("a spectrum with this much structure must be worth shaping");

        let mut quantized = [0.0f32; MAX_TNS_ORDER];
        for i in 0..filter.spec.order {
            let table: &[f32] =
                if filter.spec.resolution == 0 { &TNS_PARCOR_3 } else { &TNS_PARCOR_4 };
            quantized[i] = table[(filter.spec.coef[i] as i32 + (table.len() / 2) as i32) as usize];
        }
        let mut lpc = [0.0f32; MAX_TNS_ORDER + 1];
        parcor_to_lpc(&quantized[..filter.spec.order], &mut lpc);

        let lo = offsets[filter.spec.start_band];
        let hi = offsets[filter.spec.stop_band];
        ar_filter(&mut spectrum[lo..hi], &lpc, filter.spec.order, filter.spec.downward);

        for (i, (&want, &got)) in original.iter().zip(spectrum.iter()).enumerate() {
            let tolerance = 1e-2 * want.abs().max(1.0);
            assert!(
                (want - got).abs() <= tolerance,
                "line {i} came back as {got}, not {want}"
            );
        }
    }

    /// Filtering must actually flatten the spectrum, or it is costing bits for
    /// nothing.
    #[test]
    fn filtering_reduces_the_residual() {
        let (offsets, bands) = real_table();
        let lines = offsets[bands];
        let original: Vec<f32> = (0..lines)
            .map(|i| {
                let envelope = (-(i as f32) / 100.0).exp();
                1.0e5 * envelope * (i as f32 * 1.1).sin()
            })
            .collect();

        let mut spectrum = original.clone();
        let filter = apply(&mut spectrum, &offsets, bands, SamplingRate::Hz44100, false).unwrap();
        assert!(filter.gain > MIN_PREDICTION_GAIN);

        let lo = offsets[filter.spec.start_band];
        let hi = offsets[filter.spec.stop_band];
        let before: f64 = original[lo..hi].iter().map(|&v| (v as f64) * v as f64).sum();
        let after: f64 = spectrum[lo..hi].iter().map(|&v| (v as f64) * v as f64).sum();
        assert!(after < before, "filtering left {after} where {before} went in");
    }

    /// White noise has nothing to predict, so nothing should be transmitted.
    #[test]
    fn an_unstructured_spectrum_is_left_alone() {
        let (offsets, bands) = real_table();
        let lines = offsets[bands];
        let mut state = 12345u32;
        let spectrum: Vec<f32> = (0..lines)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1 << 23) as f32 - 1.0) * 1.0e4
            })
            .collect();

        let mut work = spectrum.clone();
        let filter = apply(&mut work, &offsets, bands, SamplingRate::Hz44100, false);
        if let Some(f) = filter {
            assert!(f.gain >= MIN_PREDICTION_GAIN, "a filter was sent for no gain");
        } else {
            assert_eq!(work, spectrum, "the spectrum was touched despite no filter");
        }
    }
}
