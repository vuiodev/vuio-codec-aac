//! High-frequency reconstruction: generating the replicated band and shaping it.
//!
//! Two stages, both operating on the QMF grid:
//!
//! 1. [`generate`] copies runs of low subbands upwards and runs each copy through
//!    a chirped second-order predictor, which flattens the tonal structure the copy
//!    carried up with it by as much as the transmitted inverse-filtering mode asks.
//! 2. [`adjust`] scales the result so each scalefactor band matches its transmitted
//!    energy, adds noise where the transmitted noise floor calls for it, and adds
//!    pure sinusoids where the encoder flagged a missing harmonic.
//!
//! # Buffer layout
//!
//! One channel's QMF grid is a band-major array of [`X_SLOTS`] slots. The newest
//! [`SLOTS_PER_FRAME`] of those are the frame just analysed; the [`HF_GEN`] slots
//! before them carry over from the previous frame, because the predictor and the
//! envelope grid both reach backwards. Output is taken from slot [`HF_ADJ`]
//! onwards, so the chain delays by `HF_GEN - HF_ADJ` slots.

use crate::decoder::sbr::data::{InverseFilterMode, SbrChannelData};
use crate::decoder::sbr::header::{BandLayout, SbrHeader};
use crate::dsp::fft::Complex32;
use crate::tables::sbr::{
    GAIN_SMOOTHING_FILTERS, LIMITER_GAINS, MAX_BOOST_GAIN, NOISE_PHASE, SINE_PHASE,
};

/// QMF slots one channel's grid holds.
pub const X_SLOTS: usize = 40;
/// Slots the QMF analysis contributes per frame.
pub const SLOTS_PER_FRAME: usize = 32;
/// Slots of look-back the predictor and the envelope grid need.
pub const HF_GEN: usize = 8;
/// Where in the grid the envelope time base and the output window begin.
pub const HF_ADJ: usize = 2;
/// Subbands the grid spans.
pub const GRID_BANDS: usize = 64;

/// Envelope energies are compared against a floor of one, which keeps a silent
/// band from producing an unbounded gain.
const ENERGY_FLOOR: f32 = 1.0;
/// Guard added to sums that could otherwise be exactly zero.
const EPS: f32 = 1e-12;
/// Guard against dividing by a noise floor of exactly zero.
const NOISE_GUARD: f32 = 1e-17;
/// Ceiling on the limiter's per-band gain, before boosting.
const MAX_LIMITER_GAIN: f32 = 1e5;
/// Slots of gain history the smoothing filter reaches back over.
const SMOOTHING_DEPTH: usize = 4;

/// Per-channel state that has to survive between frames.
#[derive(Debug, Clone)]
pub struct HfState {
    /// The QMF grid, band-major: `x[band * X_SLOTS + slot]`.
    pub x: Vec<Complex32>,
    /// Chirp factor per inverse-filtering band, smoothed across frames.
    chirp: Vec<f32>,
    /// Which subbands carried an added sinusoid last frame.
    sine_active: [bool; GRID_BANDS],
    /// Gains of the last four slots, for the smoothing filter.
    gain_history: [Vec<f32>; SMOOTHING_DEPTH],
    /// Noise levels of the last four slots.
    noise_history: [Vec<f32>; SMOOTHING_DEPTH],
    /// Position in the shared noise sequence.
    noise_phase: usize,
    /// Position in the four-step sinusoid phase.
    sine_phase: usize,
    /// Whether the first envelope of this frame also suppresses noise, because the
    /// previous frame's transient sat on its very last envelope.
    carried_transient: bool,
    /// Set until the first frame has filled the smoothing history.
    starting_up: bool,
}

impl Default for HfState {
    fn default() -> Self {
        Self::new()
    }
}

impl HfState {
    /// A cleared state.
    pub fn new() -> Self {
        Self {
            x: vec![Complex32::default(); GRID_BANDS * X_SLOTS],
            chirp: Vec::new(),
            sine_active: [false; GRID_BANDS],
            gain_history: std::array::from_fn(|_| vec![0.0; GRID_BANDS]),
            noise_history: std::array::from_fn(|_| vec![0.0; GRID_BANDS]),
            noise_phase: 0,
            sine_phase: 0,
            carried_transient: false,
            starting_up: true,
        }
    }

    /// Forget everything, as after a seek.
    pub fn reset(&mut self) {
        self.x.fill(Complex32::default());
        self.chirp.clear();
        self.sine_active = [false; GRID_BANDS];
        for row in &mut self.gain_history {
            row.fill(0.0);
        }
        for row in &mut self.noise_history {
            row.fill(0.0);
        }
        self.noise_phase = 0;
        self.sine_phase = 0;
        self.carried_transient = false;
        self.starting_up = true;
    }

    /// Value at `band`, `slot`.
    #[inline]
    pub fn at(&self, band: usize, slot: usize) -> Complex32 {
        self.x[band * X_SLOTS + slot]
    }

    /// Slots of one band, as a slice.
    #[inline]
    fn band(&self, band: usize) -> &[Complex32] {
        &self.x[band * X_SLOTS..(band + 1) * X_SLOTS]
    }

    /// Slots of one band, mutably.
    #[inline]
    fn band_mut(&mut self, band: usize) -> &mut [Complex32] {
        &mut self.x[band * X_SLOTS..(band + 1) * X_SLOTS]
    }

    /// Slide the newest [`HF_GEN`] slots down to the front, ready for the next
    /// frame's analysis to fill the rest.
    pub fn advance_frame(&mut self) {
        for band in 0..GRID_BANDS {
            let base = band * X_SLOTS;
            self.x.copy_within(base + SLOTS_PER_FRAME..base + X_SLOTS, base);
            self.x[base + HF_GEN..base + X_SLOTS].fill(Complex32::default());
        }
    }

    /// Write one analysed slot of the current frame into the grid.
    #[inline]
    pub fn store_slot(&mut self, slot: usize, bands: &[Complex32]) {
        debug_assert!(slot < SLOTS_PER_FRAME);
        for (k, &v) in bands.iter().enumerate() {
            self.x[k * X_SLOTS + HF_GEN + slot] = v;
        }
    }

    /// Read out the output window as `slot`-major subband samples.
    pub fn output_slot(&self, slot: usize, out: &mut [Complex32]) {
        debug_assert!(slot < SLOTS_PER_FRAME);
        for (k, dst) in out.iter_mut().enumerate() {
            *dst = self.x[k * X_SLOTS + HF_ADJ + slot];
        }
    }
}

/// Second-order prediction coefficients for one subband.
#[derive(Debug, Clone, Copy, Default)]
struct Predictor {
    a0: Complex32,
    a1: Complex32,
}

/// Fit a second-order predictor to each of the core's subbands.
///
/// The fit is a covariance-method solve on the whole grid, whose only role is to
/// describe how tonal the band is; the coefficients are then chirped towards zero
/// before use, which is how the decoder controls the tonality of the copy.
fn fit_predictors(state: &HfState, bands: usize) -> Vec<Predictor> {
    let mut out = vec![Predictor::default(); bands];
    for (k, predictor) in out.iter_mut().enumerate() {
        let x = state.band(k);

        let mut phi_01 = Complex32::default();
        let mut phi_02 = Complex32::default();
        let mut phi_11 = 0.0f32;
        let mut phi_12 = Complex32::default();
        let mut phi_22 = 0.0f32;

        for l in HF_ADJ..X_SLOTS {
            let (x0, x1, x2) = (x[l], x[l - 1], x[l - 2]);
            phi_01.re += x0.re * x1.re + x0.im * x1.im;
            phi_01.im += x0.im * x1.re - x0.re * x1.im;
            phi_02.re += x0.re * x2.re + x0.im * x2.im;
            phi_02.im += x0.im * x2.re - x0.re * x2.im;
            phi_11 += x1.re * x1.re + x1.im * x1.im;
            phi_12.re += x1.re * x2.re + x1.im * x2.im;
            phi_12.im += x1.im * x2.re - x1.re * x2.im;
            phi_22 += x2.re * x2.re + x2.im * x2.im;
        }

        // The 1.000001 divisor is the reference implementation's hedge against a
        // determinant that is only zero to rounding.
        let det = phi_22 * phi_11 - (phi_12.re * phi_12.re + phi_12.im * phi_12.im) / 1.000_001;

        let a1 = if det == 0.0 {
            Complex32::default()
        } else {
            let re = phi_01.re * phi_12.re - phi_01.im * phi_12.im - phi_02.re * phi_11;
            let im = phi_01.re * phi_12.im + phi_01.im * phi_12.re - phi_02.im * phi_11;
            Complex32::new(re / det, im / det)
        };

        let a0 = if phi_11 == 0.0 {
            Complex32::default()
        } else {
            let re = phi_01.re + a1.re * phi_12.re + a1.im * phi_12.im;
            let im = phi_01.im + a1.im * phi_12.re - a1.re * phi_12.im;
            Complex32::new(-re / phi_11, -im / phi_11)
        };

        // An unstable fit would ring rather than whiten; the standard discards it.
        let magnitude = |c: Complex32| c.re * c.re + c.im * c.im;
        *predictor = if magnitude(a0) >= 16.0 || magnitude(a1) >= 16.0 {
            Predictor::default()
        } else {
            Predictor { a0, a1 }
        };
    }
    out
}

/// Smooth this frame's chirp targets against the previous frame's, so that a mode
/// change does not step the tonality of a band in one slot.
fn update_chirp(state: &mut HfState, invf: &[InverseFilterMode]) {
    if state.chirp.len() != invf.len() {
        state.chirp = vec![0.0; invf.len()];
    }
    for (current, mode) in state.chirp.iter_mut().zip(invf.iter()) {
        let target = mode.chirp_target();
        // Falling towards a more tonal copy is allowed to move faster than rising
        // towards a whiter one.
        let mut next = if target < *current {
            0.75 * target + 0.25 * *current
        } else {
            0.906_25 * target + 0.093_75 * *current
        };
        if next < 0.015_625 {
            next = 0.0;
        }
        next = next.min(0.996_093_75);
        *current = next;
    }
}

/// Copy the core's subbands into the replicated range and whiten each copy.
pub fn generate(state: &mut HfState, layout: &BandLayout, data: &SbrChannelData) {
    update_chirp(state, &data.invf);
    let predictors = fit_predictors(state, layout.master[0] as usize);

    for patch in &layout.patches {
        for k in patch.dst_start..patch.dst_start + patch.width {
            let source = patch.source_of(k);
            if source >= predictors.len() {
                continue;
            }

            // Inverse filtering is transmitted per noise band; find the one that
            // covers this output subband.
            let band = layout
                .noise
                .windows(2)
                .position(|w| (k as u8) >= w[0] && (k as u8) < w[1])
                .unwrap_or(0);
            let bw = state.chirp.get(band).copied().unwrap_or(0.0);

            let predictor = predictors.get(source).copied().unwrap_or_default();
            let a0 = Complex32::new(predictor.a0.re * bw, predictor.a0.im * bw);
            let bw2 = bw * bw;
            let a1 = Complex32::new(predictor.a1.re * bw2, predictor.a1.im * bw2);

            let src_base = source * X_SLOTS;
            let dst_base = k * X_SLOTS;
            for l in HF_ADJ..X_SLOTS {
                let x0 = state.x[src_base + l];
                let x1 = state.x[src_base + l - 1];
                let x2 = state.x[src_base + l - 2];
                state.x[dst_base + l] = Complex32::new(
                    x0.re + a0.re * x1.re - a0.im * x1.im + a1.re * x2.re - a1.im * x2.im,
                    x0.im + a0.re * x1.im + a0.im * x1.re + a1.re * x2.im + a1.im * x2.re,
                );
            }
        }
    }

    // Nothing generated above the replicated range; make sure no stale content
    // from a wider previous layout survives there.
    for k in layout.k_high..GRID_BANDS {
        state.band_mut(k).fill(Complex32::default());
    }
}

/// Per-subband shaping parameters for one envelope.
struct EnvelopeGains {
    gain: Vec<f32>,
    noise: Vec<f32>,
    sine: Vec<f32>,
}

/// Scale the replicated range to its transmitted envelope, then add noise and
/// sinusoids.
pub fn adjust(
    state: &mut HfState,
    layout: &BandLayout,
    header: &SbrHeader,
    data: &SbrChannelData,
) {
    let k_x = layout.k_x;
    let width = layout.replicated_bands();
    if width == 0 {
        return;
    }

    // Which replicated subband each flagged harmonic sits in: the centre of its
    // high-resolution scalefactor band.
    let mut sine_at = vec![false; width];
    for (j, &flag) in data.add_harmonic.iter().enumerate() {
        if !flag || j + 1 >= layout.high.len() {
            continue;
        }
        let centre =
            ((layout.high[j] as usize + layout.high[j + 1] as usize) / 2).saturating_sub(k_x);
        if centre < width {
            sine_at[centre] = true;
        }
    }

    let smoothing_span = if header.smoothing_mode { 0 } else { SMOOTHING_DEPTH };
    let mut noise_env = 0usize;

    for env in 0..data.grid.envelopes() {
        // The noise floor changes at its own borders, which are a subset of the
        // envelope borders.
        while noise_env + 1 < data.grid.noise_envelopes()
            && data.grid.noise_borders[noise_env + 1] <= data.grid.borders[env]
        {
            noise_env += 1;
        }

        let transient = data.grid.transient_envelope == Some(env)
            || (env == 0 && state.carried_transient);
        let span = if transient { 0 } else { smoothing_span };

        let start = (HF_ADJ as i32 + 2 * data.grid.borders[env]).max(0) as usize;
        let end = ((HF_ADJ as i32 + 2 * data.grid.borders[env + 1]) as usize).min(X_SLOTS);
        if start >= end {
            continue;
        }

        let gains = compute_gains(
            state, layout, header, data, env, noise_env, &sine_at, transient, start, end,
        );
        apply_gains(state, layout, &gains, transient, span, start, end);
    }

    // Remember which subbands ended the frame with a sinusoid, and whether the
    // transient sat on the very last envelope, so the next frame continues both.
    state.sine_active = [false; GRID_BANDS];
    for (i, &on) in sine_at.iter().enumerate() {
        state.sine_active[k_x + i] = on;
    }
    state.carried_transient = data.grid.transient_envelope == Some(data.grid.envelopes());
    state.starting_up = false;
}

/// Work out the gain, noise level and sine level of every replicated subband for
/// one envelope.
#[allow(clippy::too_many_arguments)]
fn compute_gains(
    state: &HfState,
    layout: &BandLayout,
    header: &SbrHeader,
    data: &SbrChannelData,
    env: usize,
    noise_env: usize,
    sine_at: &[bool],
    transient: bool,
    start: usize,
    end: usize,
) -> EnvelopeGains {
    let k_x = layout.k_x;
    let width = layout.replicated_bands();
    let high_res = data.grid.high_res[env];
    let table = layout.sfb_table(high_res);
    let bands = layout.sfb_count(high_res);

    let mut reference = vec![0.0f32; width];
    let mut current = vec![0.0f32; width];
    let mut gain = vec![0.0f32; width];
    let mut noise = vec![0.0f32; width];
    let mut sine = vec![0.0f32; width];

    let envelope = data.envelope.get(env).map(|v| v.as_slice()).unwrap_or(&[]);
    let floors = data.noise.get(noise_env).map(|v| v.as_slice()).unwrap_or(&[]);
    let slots = (end - start) as f32;

    for j in 0..bands {
        let lo = table[j] as usize;
        let hi = table[j + 1] as usize;
        let transmitted = envelope.get(j).copied().unwrap_or(0.0);

        // Measured energy of the generated signal, per subband.
        for k in lo..hi {
            let i = k - k_x;
            let band = state.band(k);
            let energy: f32 =
                band[start..end].iter().map(|c| c.re * c.re + c.im * c.im).sum();
            current[i] = energy / slots;
        }
        // Without frequency interpolation the whole scalefactor band shares one
        // measurement, which makes the gain flat across it.
        if !header.interpol_freq && hi > lo {
            let mean: f32 = current[lo - k_x..hi - k_x].iter().sum::<f32>() / (hi - lo) as f32;
            current[lo - k_x..hi - k_x].fill(mean);
        }

        // A sinusoid anywhere in the band changes how the whole band is gained,
        // because the sine carries energy the gain must not also supply.
        let band_has_sine = (lo..hi).any(|k| {
            let i = k - k_x;
            sine_at[i] && (data.grid.transient_envelope.is_none_or(|t| env >= t) || state.sine_active[k])
        });

        for k in lo..hi {
            let i = k - k_x;
            let noise_band = layout
                .noise
                .windows(2)
                .position(|w| (k as u8) >= w[0] && (k as u8) < w[1])
                .unwrap_or(0);
            let q = floors.get(noise_band).copied().unwrap_or(0.0);
            let split = q / (1.0 + q + NOISE_GUARD);

            reference[i] = transmitted;
            let denominator = current[i] + ENERGY_FLOOR;

            gain[i] = if band_has_sine {
                (transmitted * split / denominator).sqrt()
            } else if transient {
                (transmitted / denominator).sqrt()
            } else {
                (transmitted * split / (denominator * (q + NOISE_GUARD))).sqrt()
            };

            if band_has_sine
                && sine_at[i]
                && (data.grid.transient_envelope.is_none_or(|t| env >= t) || state.sine_active[k])
            {
                sine[i] = (transmitted * split / (q + NOISE_GUARD)).sqrt();
            }
            noise[i] = (transmitted * split).sqrt();
        }
    }

    limit(layout, header, &reference, &current, &mut gain, &mut noise, &mut sine, transient);
    EnvelopeGains { gain, noise, sine }
}

/// Hold each limiter band's gains near the band's average, then restore the energy
/// the limiting took away.
#[allow(clippy::too_many_arguments)]
fn limit(
    layout: &BandLayout,
    header: &SbrHeader,
    reference: &[f32],
    current: &[f32],
    gain: &mut [f32],
    noise: &mut [f32],
    sine: &mut [f32],
    transient: bool,
) {
    let ceiling = LIMITER_GAINS[header.limiter_gains as usize];

    for pair in layout.limiter.windows(2) {
        let lo = pair[0] as usize;
        let hi = (pair[1] as usize).min(gain.len());
        if lo >= hi {
            continue;
        }

        let p_ref: f32 = reference[lo..hi].iter().sum();
        let p_est: f32 = current[lo..hi].iter().sum();
        let average = ((p_ref + EPS) / (p_est + EPS)).sqrt();
        let max_gain = (average * ceiling).min(MAX_LIMITER_GAIN);

        for k in lo..hi {
            if max_gain <= gain[k] {
                noise[k] *= max_gain / (gain[k] + NOISE_GUARD);
                gain[k] = max_gain;
            }
        }

        // Limiting removes energy; a bounded boost puts back what it can.
        let mut adjusted = 0.0f32;
        for k in lo..hi {
            adjusted += gain[k] * gain[k] * current[k];
            if sine[k] != 0.0 {
                adjusted += sine[k] * sine[k];
            } else if !transient {
                adjusted += noise[k] * noise[k];
            }
        }
        let boost = ((p_ref + EPS) / (adjusted + EPS)).sqrt().min(MAX_BOOST_GAIN);
        for k in lo..hi {
            gain[k] *= boost;
            noise[k] *= boost;
            sine[k] *= boost;
        }
    }
}

/// Scale the grid by the computed gains, adding noise and sinusoids as it goes.
#[allow(clippy::too_many_arguments)]
fn apply_gains(
    state: &mut HfState,
    layout: &BandLayout,
    gains: &EnvelopeGains,
    transient: bool,
    smoothing_span: usize,
    start: usize,
    end: usize,
) {
    let k_x = layout.k_x;
    let width = layout.replicated_bands();
    let filter = GAIN_SMOOTHING_FILTERS[smoothing_span];

    // The very first envelope after a reset has no history to smooth against;
    // priming it with the current gains avoids a fade-in that is not in the signal.
    if state.starting_up {
        for row in &mut state.gain_history {
            row[..width].copy_from_slice(&gains.gain[..width]);
        }
        for row in &mut state.noise_history {
            row[..width].copy_from_slice(&gains.noise[..width]);
        }
    }

    for slot in start..end {
        for i in 0..width {
            let mut smoothed_gain = 0.0f32;
            let mut smoothed_noise = 0.0f32;
            for (tap, &weight) in filter.iter().enumerate() {
                let depth = SMOOTHING_DEPTH - smoothing_span + tap;
                let (g, n) = if depth == SMOOTHING_DEPTH {
                    (gains.gain[i], gains.noise[i])
                } else {
                    (state.gain_history[depth][i], state.noise_history[depth][i])
                };
                smoothed_gain += g * weight;
                smoothed_noise += n * weight;
            }

            state.noise_phase = (state.noise_phase + 1) & 511;
            let (nr, ni) = NOISE_PHASE[state.noise_phase];
            let level = if gains.sine[i] != 0.0 || transient { 0.0 } else { smoothed_noise };

            let at = (k_x + i) * X_SLOTS + slot;
            let v = state.x[at];
            state.x[at] =
                Complex32::new(v.re * smoothed_gain + level * nr, v.im * smoothed_gain + level * ni);
        }

        // Slide the smoothing history forward one slot.
        state.gain_history.rotate_left(1);
        state.noise_history.rotate_left(1);
        state.gain_history[SMOOTHING_DEPTH - 1][..width].copy_from_slice(&gains.gain[..width]);
        state.noise_history[SMOOTHING_DEPTH - 1][..width].copy_from_slice(&gains.noise[..width]);
    }

    // Sinusoids are added after the gain, not scaled by it: their level is already
    // the level the envelope asks for.
    if gains.sine.iter().all(|&s| s == 0.0) {
        state.sine_phase = (state.sine_phase + (end - start)) & 3;
        return;
    }
    for slot in start..end {
        let (cos, sin) = SINE_PHASE[state.sine_phase];
        let mut alternating = if k_x.is_multiple_of(2) { 1.0f32 } else { -1.0 };
        for i in 0..width {
            let level = gains.sine[i];
            if level != 0.0 {
                let at = (k_x + i) * X_SLOTS + slot;
                let v = state.x[at];
                state.x[at] =
                    Complex32::new(v.re + level * cos, v.im + level * alternating * sin);
            }
            alternating = -alternating;
        }
        state.sine_phase = (state.sine_phase + 1) & 3;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::sbr::grid::FrameGrid;
    use crate::decoder::sbr::header::SbrHeader;

    fn layout() -> BandLayout {
        BandLayout::derive(&SbrHeader::default(), 44100).unwrap()
    }

    /// Filling the core bands with a tone and generating must produce energy in the
    /// replicated range, and leave the core bands untouched.
    #[test]
    fn generation_fills_the_replicated_range() {
        let layout = layout();
        let mut state = HfState::new();
        for k in 0..layout.master[0] as usize {
            for slot in 0..X_SLOTS {
                let phase = 0.3 * slot as f32 + 0.1 * k as f32;
                state.x[k * X_SLOTS + slot] = Complex32::new(phase.cos(), phase.sin());
            }
        }
        let before: Vec<Complex32> = state.x[..(layout.k_x * X_SLOTS)].to_vec();

        let data = SbrChannelData {
            invf: vec![InverseFilterMode::Off; layout.noise_band_count()],
            ..Default::default()
        };
        generate(&mut state, &layout, &data);

        assert_eq!(&state.x[..layout.k_x * X_SLOTS], &before[..], "core bands were disturbed");
        for k in layout.k_x..layout.k_high {
            let energy: f32 = state.band(k)[HF_ADJ..].iter().map(|c| c.re * c.re + c.im * c.im).sum();
            assert!(energy > 0.0, "subband {k} was left empty");
        }
        for k in layout.k_high..GRID_BANDS {
            assert!(state.band(k).iter().all(|c| c.re == 0.0 && c.im == 0.0));
        }
    }

    /// With inverse filtering off the copy must be exact, so a patched subband
    /// reproduces its source.
    #[test]
    fn a_patch_with_no_whitening_copies_exactly() {
        let layout = layout();
        let mut state = HfState::new();
        for k in 0..layout.master[0] as usize {
            for slot in 0..X_SLOTS {
                let v = (k * 7 + slot * 3) as f32;
                state.x[k * X_SLOTS + slot] = Complex32::new(v.sin(), v.cos());
            }
        }
        let source_copy = state.x.clone();

        let data = SbrChannelData {
            invf: vec![InverseFilterMode::Off; layout.noise_band_count()],
            ..Default::default()
        };
        generate(&mut state, &layout, &data);

        let patch = layout.patches[0];
        for k in patch.dst_start..patch.dst_start + patch.width {
            let source = patch.source_of(k);
            for slot in HF_ADJ..X_SLOTS {
                let got = state.x[k * X_SLOTS + slot];
                let want = source_copy[source * X_SLOTS + slot];
                assert!(
                    (got.re - want.re).abs() < 1e-5 && (got.im - want.im).abs() < 1e-5,
                    "band {k} slot {slot}: {got:?} != {want:?}"
                );
            }
        }
    }

    /// Adjustment must drive each scalefactor band to its transmitted energy.
    #[test]
    fn adjustment_hits_the_transmitted_energy() {
        let layout = layout();
        let mut state = HfState::new();
        for k in layout.k_x..layout.k_high {
            for slot in 0..X_SLOTS {
                // Well above the energy floor, so the `+ 1` guard does not bite.
                state.x[k * X_SLOTS + slot] = Complex32::new(20.0, -10.0);
            }
        }

        let high_bands = layout.sfb_count(true);
        let target = 4000.0f32;
        let mut data = SbrChannelData {
            grid: FrameGrid::default(),
            invf: vec![InverseFilterMode::Off; layout.noise_band_count()],
            envelope: vec![vec![target; high_bands]],
            // A vanishing noise floor asks for all of the energy as signal.
            noise: vec![vec![1e-6; layout.noise_band_count()]],
            add_harmonic: vec![false; high_bands],
            ..Default::default()
        };
        data.grid.high_res = vec![true];
        let header = SbrHeader { limiter_gains: 3, ..SbrHeader::default() };

        adjust(&mut state, &layout, &header, &data);

        for j in 0..high_bands {
            let lo = layout.high[j] as usize;
            let hi = layout.high[j + 1] as usize;
            let slots = 2 * data.grid.borders[1] as usize;
            let mut energy = 0.0f64;
            for k in lo..hi {
                for slot in HF_ADJ..HF_ADJ + slots {
                    let c = state.x[k * X_SLOTS + slot];
                    energy += (c.re * c.re + c.im * c.im) as f64;
                }
            }
            let mean = energy / (slots * (hi - lo)) as f64;
            assert!(
                (mean / target as f64 - 1.0).abs() < 0.02,
                "band {j}: energy {mean:.1} is not near the transmitted {target}"
            );
        }
    }

    /// Advancing must carry the newest slots down and clear the rest.
    #[test]
    fn advancing_keeps_the_tail() {
        let mut state = HfState::new();
        for k in 0..GRID_BANDS {
            for slot in 0..X_SLOTS {
                state.x[k * X_SLOTS + slot] = Complex32::new(slot as f32, k as f32);
            }
        }
        state.advance_frame();
        for k in 0..GRID_BANDS {
            for slot in 0..HF_GEN {
                assert_eq!(state.at(k, slot), Complex32::new((slot + SLOTS_PER_FRAME) as f32, k as f32));
            }
            for slot in HF_GEN..X_SLOTS {
                assert_eq!(state.at(k, slot), Complex32::default());
            }
        }
    }
}
