//! Temporal Noise Shaping (TNS).
//!
//! TNS shapes quantization noise inside a frame by running a linear-prediction
//! filter *along the frequency axis*. The encoder applies the analysis (all-zero)
//! filter; the decoder applies the matching synthesis (all-pole) filter.
//!
//! Filters are transmitted as quantized reflection (PARCOR) coefficients, which are
//! converted to direct-form LPC before filtering. See ISO/IEC 14496-3 clause 4.6.9
//! and `decoder/ixheaacd_aac_tns.c`.

use crate::decoder::aac::ics::{ChannelData, IcsInfo, MAX_TNS_ORDER, TnsFilterSpec};
use crate::tables::sfb::TNS_MAX_BANDS;
use crate::tables::scalefactor::sampling_rate_index;
use crate::types::SamplingRate;

/// Reflection coefficients for 3-bit resolution, indexed by `coef + 4`.
///
/// These are `sin(c / iqfac)` with `iqfac = 3.5 / (pi/2)` for non-negative `c` and
/// `4.5 / (pi/2)` for negative `c`, matching `tns_coeff3` in the reference ROM.
pub static TNS_PARCOR_3: [f32; 8] = [
    -0.984_807_75,
    -0.866_025_4,
    -0.642_787_6,
    -0.342_020_15,
    0.0,
    0.433_883_75,
    0.781_831_5,
    0.974_927_9,
];

/// Reflection coefficients for 4-bit resolution, indexed by `coef + 8`.
pub static TNS_PARCOR_4: [f32; 16] = [
    -0.995_734_2,
    -0.961_825_6,
    -0.895_163_3,
    -0.798_017_2,
    -0.673_695_6,
    -0.526_432_2,
    -0.361_241_67,
    -0.183_749_52,
    0.0,
    0.207_911_69,
    0.406_736_65,
    0.587_785_25,
    0.743_144_8,
    0.866_025_4,
    0.951_056_5,
    0.994_521_9,
];

/// Convert reflection coefficients to direct-form LPC coefficients.
///
/// Returns `order + 1` taps with `lpc[0] == 1`. The recursion is the standard
/// step-up (Levinson) form: at stage `m`, `a[i] += k * a[m - i]` and `a[m] = k`.
#[inline]
pub fn parcor_to_lpc(parcor: &[f32], lpc: &mut [f32; MAX_TNS_ORDER + 1]) {
    let order = parcor.len();
    lpc[0] = 1.0;
    let mut work = [0.0f32; MAX_TNS_ORDER + 1];

    for m in 1..=order {
        let k = parcor[m - 1];
        for i in 1..m {
            work[i] = lpc[i] + k * lpc[m - i];
        }
        lpc[1..m].copy_from_slice(&work[1..m]);
        lpc[m] = k;
    }
}

/// Apply the all-pole synthesis filter in place over `spectrum`.
///
/// `descending` runs the filter from the high-frequency end downward, which is what
/// a filter with its direction bit set signals.
#[inline]
pub fn ar_filter(spectrum: &mut [f32], lpc: &[f32], order: usize, descending: bool) {
    if order == 0 || spectrum.is_empty() {
        return;
    }
    let mut state = [0.0f32; MAX_TNS_ORDER + 1];

    if descending {
        for i in (0..spectrum.len()).rev() {
            let mut y = spectrum[i];
            for j in (1..=order).rev() {
                y -= lpc[j] * state[j - 1];
                state[j] = state[j - 1];
            }
            state[0] = y;
            spectrum[i] = y;
        }
    } else {
        for sample in spectrum.iter_mut() {
            let mut y = *sample;
            for j in (1..=order).rev() {
                y -= lpc[j] * state[j - 1];
                state[j] = state[j - 1];
            }
            state[0] = y;
            *sample = y;
        }
    }
}

/// Apply the all-zero analysis filter in place; the exact inverse of [`ar_filter`].
///
/// Only the encoder needs this, but keeping the pair together makes the inverse
/// relationship testable.
#[inline]
pub fn ma_filter(spectrum: &mut [f32], lpc: &[f32], order: usize, descending: bool) {
    if order == 0 || spectrum.is_empty() {
        return;
    }
    let mut state = [0.0f32; MAX_TNS_ORDER + 1];

    let mut step = |x: &mut f32| {
        let mut y = *x;
        for j in (1..=order).rev() {
            y += lpc[j] * state[j - 1];
            state[j] = state[j - 1];
        }
        state[0] = *x;
        *x = y;
    };

    if descending {
        for i in (0..spectrum.len()).rev() {
            step(&mut spectrum[i]);
        }
    } else {
        for sample in spectrum.iter_mut() {
            step(sample);
        }
    }
}

/// Resolve one filter's PARCOR indices into reflection coefficients.
#[inline]
fn filter_parcor(filter: &TnsFilterSpec, out: &mut [f32; MAX_TNS_ORDER]) {
    if filter.resolution == 0 {
        for i in 0..filter.order {
            let idx = (filter.coef[i] as i32 + 4).clamp(0, 7) as usize;
            out[i] = TNS_PARCOR_3[idx];
        }
    } else {
        for i in 0..filter.order {
            let idx = (filter.coef[i] as i32 + 8).clamp(0, 15) as usize;
            out[i] = TNS_PARCOR_4[idx];
        }
    }
}

/// Highest scalefactor band TNS may touch at this rate and window length.
#[inline]
pub fn tns_max_band(rate: SamplingRate, is_short: bool) -> usize {
    let idx = sampling_rate_index(rate.hz()).min(TNS_MAX_BANDS.len() - 1);
    TNS_MAX_BANDS[idx][if is_short { 1 } else { 0 }] as usize
}

/// Apply every TNS filter of a channel to its per-window spectrum.
///
/// `spec` holds `num_windows` consecutive windows of `ics.window_length` lines, i.e.
/// the deinterleaved layout.
pub fn apply_tns(ch: &mut ChannelData, rate: SamplingRate) {
    if !ch.tns.present {
        return;
    }
    let ics: IcsInfo = ch.ics.clone();
    let is_short = ics.window_sequence.is_eight_short();
    let max_band = tns_max_band(rate, is_short).min(ics.max_sfb).min(ics.num_swb);

    for w in 0..ics.num_windows {
        let win_start = w * ics.window_length;
        for f in 0..ch.tns.n_filt[w] {
            let filter = ch.tns.filters[w][f];
            if filter.order == 0 {
                continue;
            }
            let start = filter.start_band.min(max_band);
            let stop = filter.stop_band.min(max_band);
            if stop <= start {
                continue;
            }

            let lo = win_start + ics.swb_offset[start] as usize;
            let hi = win_start + ics.swb_offset[stop] as usize;
            if hi > ch.spec.len() || hi <= lo {
                continue;
            }

            let mut parcor = [0.0f32; MAX_TNS_ORDER];
            filter_parcor(&filter, &mut parcor);
            let mut lpc = [0.0f32; MAX_TNS_ORDER + 1];
            parcor_to_lpc(&parcor[..filter.order], &mut lpc);

            ar_filter(&mut ch.spec[lo..hi], &lpc, filter.order, filter.downward);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reflection-coefficient tables must match the quantizer the standard
    /// defines, and by extension the reference ROM they were checked against.
    #[test]
    fn parcor_tables_match_the_quantizer() {
        let half_pi = std::f64::consts::FRAC_PI_2;
        for (res, table) in [(3usize, &TNS_PARCOR_3[..]), (4, &TNS_PARCOR_4[..])] {
            let half = 1i32 << (res - 1);
            let iqfac = (half as f64 - 0.5) / half_pi;
            let iqfac_m = (half as f64 + 0.5) / half_pi;
            for (i, &got) in table.iter().enumerate() {
                let c = i as i32 - half;
                let a = if c >= 0 { c as f64 / iqfac } else { c as f64 / iqfac_m };
                let want = a.sin();
                assert!(
                    (got as f64 - want).abs() < 1e-6,
                    "res {res} index {i}: {got} vs {want}"
                );
            }
        }
    }

    /// Our tables must equal the reference decoder's Q31 tables to f32 precision.
    #[test]
    fn parcor_tables_match_reference_q31() {
        let c3: [i32; 8] = [
            -2114858540, -1859775364, -1380375817, -734482679, 0, 931758215, 1678970362,
            2093641723,
        ];
        let c4: [i32; 16] = [
            -2138322869, -2065504899, -1922348549, -1713729017, -1446750457, -1130504584,
            -775760644, -394599111, 0, 446486976, 873460283, 1262259191, 1595891328, 1859775364,
            2042378368, 2135719561,
        ];
        for (got, raw) in TNS_PARCOR_3.iter().zip(c3.iter()) {
            let want = *raw as f64 / 2147483648.0;
            assert!((*got as f64 - want).abs() < 1e-6, "{got} vs {want}");
        }
        for (got, raw) in TNS_PARCOR_4.iter().zip(c4.iter()) {
            let want = *raw as f64 / 2147483648.0;
            assert!((*got as f64 - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    /// `lpc[0]` is always 1 and `lpc[order]` is always the last reflection
    /// coefficient, for every order.
    #[test]
    fn lpc_conversion_endpoints() {
        for order in 1..=MAX_TNS_ORDER {
            let parcor: Vec<f32> = (0..order).map(|i| 0.4 * ((i % 5) as f32 - 2.0) / 2.0).collect();
            let mut lpc = [0.0f32; MAX_TNS_ORDER + 1];
            parcor_to_lpc(&parcor, &mut lpc);
            assert_eq!(lpc[0], 1.0);
            assert!((lpc[order] - parcor[order - 1]).abs() < 1e-6);
        }
    }

    /// The analysis and synthesis filters must invert each other exactly, which is
    /// the property TNS relies on end to end.
    #[test]
    fn ar_and_ma_filters_are_inverses() {
        for &order in &[1usize, 2, 3, 7, 12, MAX_TNS_ORDER] {
            for &descending in &[false, true] {
                let parcor: Vec<f32> =
                    (0..order).map(|i| 0.6 * (((i * 7 % 11) as f32 / 11.0) - 0.5)).collect();
                let mut lpc = [0.0f32; MAX_TNS_ORDER + 1];
                parcor_to_lpc(&parcor, &mut lpc);

                let original: Vec<f32> =
                    (0..96).map(|i| ((i as f32) * 0.31).sin() * 100.0).collect();
                let mut work = original.clone();

                ma_filter(&mut work, &lpc, order, descending);
                ar_filter(&mut work, &lpc, order, descending);

                for (i, (&a, &b)) in original.iter().zip(work.iter()).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-2,
                        "order {order} desc {descending} line {i}: {a} vs {b}"
                    );
                }
            }
        }
    }

    /// A zero-order filter must leave the spectrum untouched.
    #[test]
    fn zero_order_is_a_no_op() {
        let mut spec: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let lpc = [1.0f32; MAX_TNS_ORDER + 1];
        ar_filter(&mut spec, &lpc, 0, false);
        assert_eq!(spec, (0..32).map(|i| i as f32).collect::<Vec<_>>());
    }

    /// Every quantized reflection coefficient must satisfy |k| < 1, which is the
    /// condition that makes the all-pole synthesis filter stable.
    #[test]
    fn all_reflection_coefficients_are_inside_the_unit_circle() {
        for &k in TNS_PARCOR_3.iter().chain(TNS_PARCOR_4.iter()) {
            assert!(k.abs() < 1.0, "reflection coefficient {k} is not stable");
        }
    }

    /// Filters built from any quantizer output must produce finite results.
    ///
    /// Reflection coefficients near +/-1 give a legitimately high-gain filter, so
    /// this checks finiteness rather than a magnitude bound; the bound belongs on
    /// realistic coefficient sets, covered below.
    #[test]
    fn synthesis_filter_stays_finite_for_all_quantized_coefficients() {
        for res in [0u8, 1u8] {
            let n = if res == 0 { 8 } else { 16 };
            for code in 0..n {
                let order = 8;
                let mut filter = TnsFilterSpec { order, resolution: res, ..Default::default() };
                let bias = if res == 0 { 4 } else { 8 };
                for i in 0..order {
                    filter.coef[i] = (code as i32 - bias) as i8;
                }
                let mut parcor = [0.0f32; MAX_TNS_ORDER];
                filter_parcor(&filter, &mut parcor);
                let mut lpc = [0.0f32; MAX_TNS_ORDER + 1];
                parcor_to_lpc(&parcor[..order], &mut lpc);

                let mut spec = vec![1.0f32; 256];
                ar_filter(&mut spec, &lpc, order, false);
                assert!(
                    spec.iter().all(|v| v.is_finite()),
                    "res {res} code {code} produced a non-finite spectrum"
                );
            }
        }
    }

    /// A mixed, encoder-realistic coefficient set must keep the filter gain modest.
    #[test]
    fn synthesis_gain_is_bounded_for_realistic_filters() {
        let order = 8;
        let mut filter = TnsFilterSpec { order, resolution: 1, ..Default::default() };
        // A decaying alternating set, typical of a real TNS analysis result.
        for i in 0..order {
            let sign = if i % 2 == 0 { 1 } else { -1 };
            filter.coef[i] = (sign * (4 - i as i32 / 2)) as i8;
        }
        let mut parcor = [0.0f32; MAX_TNS_ORDER];
        filter_parcor(&filter, &mut parcor);
        let mut lpc = [0.0f32; MAX_TNS_ORDER + 1];
        parcor_to_lpc(&parcor[..order], &mut lpc);

        let mut spec = vec![1.0f32; 256];
        ar_filter(&mut spec, &lpc, order, false);
        let peak = spec.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak < 100.0, "realistic filter gain {peak} is implausibly high");
    }
}
