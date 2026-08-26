//! Complex quadrature mirror filterbanks for SBR, PS and MPEG Surround.
//!
//! Three banks live here, all built on the single 640-tap prototype in
//! [`crate::tables::qmf`]:
//!
//! * [`QmfAnalysis`] — 32 complex subbands per 32 input samples. This is what turns
//!   the AAC core signal into the time/frequency grid SBR operates on.
//! * [`QmfSynthesis`] — 64 complex subbands back to 64 samples, doubling the sample
//!   rate. This closes the SBR chain.
//! * [`QmfSynthesis`] in *downsampled* mode — 32 subbands back to 32 samples, for
//!   streams where SBR is signalled but the decoder is asked for core-rate output.
//!
//! # Why an FFT
//!
//! Written out, the modulation of the analysis bank is
//!
//! ```text
//! X[k] = sum over n in 0..64 of u[n] * exp(i*pi/64 * (k + 1/2) * (2n - 31))
//! ```
//!
//! which is 32x64 complex multiply-accumulates per time slot, and the synthesis
//! bank is worse at 64x128. Folding the half-sample offsets into a pre- and a
//! post-twiddle leaves a plain DFT in the middle, so each bank costs one power-of-two
//! FFT per slot instead: 64-point for analysis, 128-point for synthesis. Over a
//! 1024-sample frame that is the difference between roughly 400k and 40k multiplies
//! per channel.
//!
//! # Buffering
//!
//! Both banks keep their delay line as a double-length ring written at two offsets,
//! so advancing a slot is a pair of writes rather than a `memmove` of the whole
//! history. The windowing pass then always reads one contiguous run.

use crate::dsp::fft::{Complex32, FftContext};
use crate::tables::qmf::{QMF_ANALYSIS_WINDOW, QMF_SYNTHESIS_WINDOW};

/// Subbands the analysis bank produces per time slot.
pub const QMF_ANALYSIS_BANDS: usize = 32;
/// Subbands the synthesis bank consumes per time slot.
pub const QMF_SYNTHESIS_BANDS: usize = 64;
/// Samples of history the analysis bank windows.
const ANALYSIS_HISTORY: usize = 320;
/// Samples of history the synthesis bank windows.
const SYNTHESIS_HISTORY: usize = 1280;

/// 32-band complex analysis filterbank.
///
/// Feeds one time slot at a time: 32 real input samples in, 32 complex subbands out.
/// An AAC frame of 1024 samples is 32 slots.
#[derive(Clone)]
pub struct QmfAnalysis {
    /// Delay line, written twice so the windowing pass never wraps.
    history: Box<[f32; 2 * ANALYSIS_HISTORY]>,
    /// Write cursor into the first copy, counting down by 32 each slot.
    cursor: usize,
    fft: FftContext,
    /// `exp(i*pi*n/64)` for `n` in `0..64`, the pre-twiddle.
    pre: Box<[Complex32; 64]>,
    /// `exp(-i*31*pi*(k + 1/2)/64)` for `k` in `0..32`, the post-twiddle.
    post: Box<[Complex32; QMF_ANALYSIS_BANDS]>,
    scratch_in: Box<[Complex32; 64]>,
    scratch_out: Box<[Complex32; 64]>,
}

impl Default for QmfAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl QmfAnalysis {
    /// Build an analysis bank with a cleared delay line.
    pub fn new() -> Self {
        let mut pre = Box::new([Complex32::default(); 64]);
        for (n, p) in pre.iter_mut().enumerate() {
            let a = std::f64::consts::PI * n as f64 / 64.0;
            *p = Complex32::new(a.cos() as f32, a.sin() as f32);
        }
        let mut post = Box::new([Complex32::default(); QMF_ANALYSIS_BANDS]);
        for (k, p) in post.iter_mut().enumerate() {
            let a = -31.0 * std::f64::consts::PI * (k as f64 + 0.5) / 64.0;
            *p = Complex32::new(a.cos() as f32, a.sin() as f32);
        }
        Self {
            history: Box::new([0.0; 2 * ANALYSIS_HISTORY]),
            cursor: ANALYSIS_HISTORY,
            fft: FftContext::new(64),
            pre,
            post,
            scratch_in: Box::new([Complex32::default(); 64]),
            scratch_out: Box::new([Complex32::default(); 64]),
        }
    }

    /// Forget all history, as after a seek.
    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.cursor = ANALYSIS_HISTORY;
    }

    /// Analyse one time slot: 32 real samples to 32 complex subbands.
    pub fn process_slot(&mut self, input: &[f32], output: &mut [Complex32]) {
        debug_assert_eq!(input.len(), QMF_ANALYSIS_BANDS);
        debug_assert_eq!(output.len(), QMF_ANALYSIS_BANDS);

        // The standard stores the incoming slot in reverse, newest sample first.
        // Writing it into both copies of the ring keeps the window read contiguous.
        if self.cursor == 0 {
            self.cursor = ANALYSIS_HISTORY;
        }
        self.cursor -= QMF_ANALYSIS_BANDS;
        let at = self.cursor;
        for (i, &s) in input.iter().enumerate() {
            let v = s;
            self.history[at + QMF_ANALYSIS_BANDS - 1 - i] = v;
            self.history[at + ANALYSIS_HISTORY + QMF_ANALYSIS_BANDS - 1 - i] = v;
        }

        // Window against the decimated prototype and fold the 320 taps into 64.
        let window = &self.history[at..at + ANALYSIS_HISTORY];
        let mut u = [0.0f32; 64];
        for j in 0..5 {
            let w = &window[j * 64..j * 64 + 64];
            let c = &QMF_ANALYSIS_WINDOW[j * 64..j * 64 + 64];
            for n in 0..64 {
                u[n] += w[n] * c[n];
            }
        }

        // X[k] = post[k] * sum_n (u[n] * pre[n]) * exp(+i*2*pi*k*n/64).
        // The inner sum is an unnormalized inverse DFT, which is the forward
        // transform of the conjugated input, conjugated again on the way out.
        for (dst, (&un, &p)) in self.scratch_in.iter_mut().zip(u.iter().zip(self.pre.iter())) {
            *dst = Complex32::new(un * p.re, -un * p.im);
        }
        self.fft.forward_into(&self.scratch_in[..], &mut self.scratch_out[..]);

        for (k, out) in output.iter_mut().enumerate() {
            let s = self.scratch_out[k];
            let t = self.post[k];
            // conj(s) * t
            *out = Complex32::new(s.re * t.re + s.im * t.im, s.re * t.im - s.im * t.re);
        }
    }

    /// Analyse a whole frame: `input.len() / 32` slots into a slot-major grid.
    ///
    /// `output[slot][band]` is the layout the SBR envelope stages want.
    pub fn process_frame(&mut self, input: &[f32], output: &mut [[Complex32; QMF_ANALYSIS_BANDS]]) {
        debug_assert_eq!(input.len(), output.len() * QMF_ANALYSIS_BANDS);
        for (slot, out) in output.iter_mut().enumerate() {
            let lo = slot * QMF_ANALYSIS_BANDS;
            self.process_slot(&input[lo..lo + QMF_ANALYSIS_BANDS], &mut out[..]);
        }
    }
}

/// How many subbands a synthesis bank runs, which sets its output rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthesisWidth {
    /// 64 subbands to 64 samples: the normal SBR output, at twice the core rate.
    Full,
    /// 32 subbands to 32 samples: core-rate output for downsampled SBR.
    Downsampled,
}

impl SynthesisWidth {
    /// Subbands consumed, and samples produced, per time slot.
    #[inline]
    pub const fn bands(self) -> usize {
        match self {
            Self::Full => 64,
            Self::Downsampled => 32,
        }
    }
}

/// Complex synthesis filterbank, 64 or 32 subbands wide.
///
/// This is the adjoint of the analysis bank built on the same prototype, which for
/// a filterbank twice oversampled — 64 complex subbands carrying 64 real samples —
/// is also its inverse. Chaining a [`QmfAnalysis`] into a [`SynthesisWidth::Full`]
/// bank reconstructs the input at twice the rate with a flat response across every
/// band; the constant `1 / 32` below is what makes the round trip unity gain.
#[derive(Clone)]
pub struct QmfSynthesis {
    width: SynthesisWidth,
    /// Ten slots of modulated samples, written twice so windowing never wraps.
    history: Box<[f32; 2 * SYNTHESIS_HISTORY]>,
    /// Length of one copy of the ring, `20 * bands`.
    span: usize,
    /// Write cursor into the first copy, counting down by `2 * bands` each slot.
    cursor: usize,
    fft: FftContext,
    /// Pre-twiddle folding the modulation's half-band and phase offsets, per subband.
    pre: Vec<Complex32>,
    /// Post-twiddle, per modulated sample.
    post: Vec<Complex32>,
    scratch_in: Vec<Complex32>,
    scratch_out: Vec<Complex32>,
}

/// Round-trip gain of an analysis bank feeding a synthesis bank of either width.
///
/// The prototype is normalised so the two banks together multiply by `2^5`; the
/// synthesis divides it back out so a decoder can hand the filterbank a signal and
/// get the same signal back.
const QMF_ROUND_TRIP_GAIN: f32 = 32.0;

impl QmfSynthesis {
    /// Build a synthesis bank of the given width, with a cleared delay line.
    pub fn new(width: SynthesisWidth) -> Self {
        let bands = width.bands();
        let two_bands = 2 * bands;
        // The modulation is
        //     W[m] = sum_k X[k] * exp(-i*pi*(k + 1/2)*(2m - (bands - 1)) / (2*bands))
        // and splitting the exponent into a term in `k*m`, one in `k` and one in `m`
        // leaves a plain `2*bands`-point DFT between a pre- and a post-twiddle.
        let n = two_bands as f64;
        let pre = (0..bands)
            .map(|k| {
                // A 32-band analysis feeding a 64-band synthesis leaves the two
                // banks' modulation references half an output sample apart. Taking
                // that half sample out here makes the doubling chain's delay the
                // round number 577 rather than 576.5, so the low bands land on
                // integer sample positions.
                let extra = match width {
                    SynthesisWidth::Full => -std::f64::consts::PI * (k as f64 + 0.5) / n,
                    SynthesisWidth::Downsampled => 0.0,
                };
                let a = std::f64::consts::PI * (bands - 1) as f64 * k as f64 / n + extra;
                Complex32::new(a.cos() as f32, a.sin() as f32)
            })
            .collect();
        let post = (0..two_bands)
            .map(|m| {
                let a = std::f64::consts::PI * (bands - 1) as f64 / (2.0 * n)
                    - std::f64::consts::PI * m as f64 / n;
                Complex32::new(a.cos() as f32, a.sin() as f32)
            })
            .collect();
        Self {
            width,
            history: Box::new([0.0; 2 * SYNTHESIS_HISTORY]),
            span: 10 * two_bands,
            cursor: 10 * two_bands,
            fft: FftContext::new(two_bands),
            pre,
            post,
            scratch_in: vec![Complex32::default(); two_bands],
            scratch_out: vec![Complex32::default(); two_bands],
        }
    }

    /// Forget all history, as after a seek.
    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.cursor = self.span;
    }

    /// Subbands consumed, and samples produced, per time slot.
    #[inline]
    pub const fn bands(&self) -> usize {
        self.width.bands()
    }

    /// Synthesise one time slot: `bands` complex subbands to `bands` real samples.
    pub fn process_slot(&mut self, input: &[Complex32], output: &mut [f32]) {
        let bands = self.bands();
        let two_bands = 2 * bands;
        debug_assert_eq!(input.len(), bands);
        debug_assert_eq!(output.len(), bands);

        for (dst, (&x, &p)) in self.scratch_in.iter_mut().zip(input.iter().zip(self.pre.iter())) {
            *dst = Complex32::new(x.re * p.re - x.im * p.im, x.re * p.im + x.im * p.re);
        }
        // Only the lower half carries subbands; the transform is over 2 * bands.
        self.scratch_in[bands..].fill(Complex32::default());
        self.fft.forward_into(&self.scratch_in, &mut self.scratch_out);

        if self.cursor == 0 {
            self.cursor = self.span;
        }
        self.cursor -= two_bands;
        let at = self.cursor;
        for m in 0..two_bands {
            let s = self.scratch_out[m];
            let t = self.post[m];
            let v = s.re * t.re - s.im * t.im;
            self.history[at + m] = v;
            self.history[at + self.span + m] = v;
        }

        // Overlap-add against the prototype. Indices run backwards in the output,
        // so the loop is written over the reversed index: that way both the delay
        // line and the window are read forwards, one contiguous run per tap block,
        // which is what the autovectoriser needs.
        let proto: &[f32] = match self.width {
            SynthesisWidth::Full => &QMF_SYNTHESIS_WINDOW,
            SynthesisWidth::Downsampled => &QMF_ANALYSIS_WINDOW,
        };
        let v = &self.history[at..at + self.span];
        let mut acc = [0.0f32; QMF_SYNTHESIS_BANDS];
        let acc = &mut acc[..bands];
        for e in 0..10 {
            let block = &v[e * two_bands..(e + 1) * two_bands];
            let half = if e % 2 == 0 { &block[bands..] } else { &block[..bands] };
            let c = &proto[(9 - e) * bands..(9 - e) * bands + bands];
            for i in 0..bands {
                acc[i] += half[i] * c[i];
            }
        }
        let scale = 1.0 / QMF_ROUND_TRIP_GAIN;
        for (i, out) in output.iter_mut().enumerate() {
            *out = acc[bands - 1 - i] * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Push a signal through an analysis bank into a synthesis bank of the given
    /// width, returning the reconstructed samples.
    fn round_trip(input: &[f32], width: SynthesisWidth) -> Vec<f32> {
        let mut analysis = QmfAnalysis::new();
        let mut synthesis = QmfSynthesis::new(width);
        let bands = width.bands();
        let slots = input.len() / QMF_ANALYSIS_BANDS;
        let mut output = vec![0.0f32; slots * bands];
        let mut low = [Complex32::default(); QMF_ANALYSIS_BANDS];
        let mut wide = [Complex32::default(); QMF_SYNTHESIS_BANDS];

        for slot in 0..slots {
            let lo = slot * QMF_ANALYSIS_BANDS;
            analysis.process_slot(&input[lo..lo + QMF_ANALYSIS_BANDS], &mut low);
            wide[..QMF_ANALYSIS_BANDS].copy_from_slice(&low);
            wide[QMF_ANALYSIS_BANDS..].fill(Complex32::default());
            synthesis.process_slot(&wide[..bands], &mut output[slot * bands..(slot + 1) * bands]);
        }
        output
    }

    /// Signal-to-noise of a reconstructed tone against the ideal, where `up` output
    /// samples correspond to one input sample and the chain delays by `delay`.
    fn tone_snr(band: f32, width: SynthesisWidth, delay: f64) -> f64 {
        let up = match width {
            SynthesisWidth::Full => 2.0f64,
            SynthesisWidth::Downsampled => 1.0,
        };
        let freq = std::f32::consts::PI * (band + 0.5) / 32.0;
        let input: Vec<f32> = (0..160 * 32).map(|i| (freq * i as f32).sin()).collect();
        let output = round_trip(&input, width);
        let w = freq as f64 / up;

        let (lo, hi) = (2000, output.len() - 2000);
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for j in lo..hi {
            let want = (w * (j as f64 - delay)).sin();
            let got = output[j] as f64;
            num += (got - want) * (got - want);
            den += want * want;
        }
        10.0 * (den / num.max(1e-30)).log10()
    }

    /// The doubling chain must reconstruct every subband transparently, at band
    /// centres and at the boundaries between bands alike.
    ///
    /// Boundaries are the demanding case: a tone there is split across two adjacent
    /// subbands, and the aliasing each one picks up only cancels if the two banks
    /// agree on the polyphase fold's sign. They did not in an earlier revision, and
    /// band centres alone did not show it.
    #[test]
    fn doubling_chain_is_transparent() {
        for band in [0.0f32, 0.5, 3.0, 8.5, 16.0, 23.5, 30.0] {
            let snr = tone_snr(band, SynthesisWidth::Full, 577.0);
            assert!(snr > 50.0, "band {band} reconstructs at only {snr:.1} dB");
        }
    }

    /// The core-rate chain must be transparent over the same sweep.
    #[test]
    fn core_rate_chain_is_transparent() {
        for band in [0.0f32, 0.5, 3.0, 8.5, 16.0, 23.5, 30.0] {
            let snr = tone_snr(band, SynthesisWidth::Downsampled, 288.0);
            assert!(snr > 50.0, "band {band} reconstructs at only {snr:.1} dB");
        }
    }

    /// Round-trip gain must be one in every subband, which is what makes the
    /// normalisation exact rather than merely close.
    #[test]
    fn round_trip_gain_is_flat_across_bands() {
        for width in [SynthesisWidth::Full, SynthesisWidth::Downsampled] {
            for band in [1usize, 5, 11, 17, 23, 29] {
                let freq = std::f32::consts::PI * (band as f32 + 0.5) / 32.0;
                let input: Vec<f32> = (0..120 * 32).map(|i| (freq * i as f32).sin()).collect();
                let output = round_trip(&input, width);

                let start = output.len() / 3;
                let n = output.len() / 2;
                let energy: f64 = output[start..start + n]
                    .iter()
                    .map(|&v| (v as f64) * (v as f64))
                    .sum::<f64>()
                    / n as f64;
                // A unit sine has mean square 1/2.
                let gain = (energy * 2.0).sqrt();
                assert!(
                    (gain - 1.0).abs() < 0.02,
                    "{width:?} band {band} round-trip gain {gain:.4} is not unity"
                );
            }
        }
    }

    /// A pure tone must land in the subband its frequency selects.
    #[test]
    fn tone_lands_in_one_band() {
        let mut analysis = QmfAnalysis::new();
        // Centre of band 7 is at (7 + 1/2)/64 of the sample rate.
        let f = 7.5 / 64.0;
        let mut bands = [Complex32::default(); 32];
        let mut energy = [0.0f64; 32];

        for slot in 0..96 {
            let chunk: Vec<f32> = (0..32)
                .map(|i| {
                    let t = (slot * 32 + i) as f32;
                    (2.0 * std::f32::consts::PI * f * t).sin()
                })
                .collect();
            analysis.process_slot(&chunk, &mut bands);
            if slot >= 20 {
                for (e, b) in energy.iter_mut().zip(bands.iter()) {
                    *e += (b.re * b.re + b.im * b.im) as f64;
                }
            }
        }

        let peak = energy.iter().cloned().fold(f64::MIN, f64::max);
        assert_eq!(
            energy.iter().position(|&e| e == peak),
            Some(7),
            "tone did not land in band 7: {energy:?}"
        );
        // Bands overlap by design, so the immediate neighbours carry a fraction of
        // the tone; three bands out and beyond must be in the stopband.
        let far: f64 = energy
            .iter()
            .enumerate()
            .filter(|&(k, _)| k.abs_diff(7) > 2)
            .map(|(_, e)| *e)
            .sum();
        assert!(far < peak * 5e-3, "stopband leakage {far} is too high");
    }

    /// Resetting must return both banks to their initial state.
    #[test]
    fn reset_clears_history() {
        let mut analysis = QmfAnalysis::new();
        let mut synthesis = QmfSynthesis::new(SynthesisWidth::Full);
        let mut bands = [Complex32::default(); 32];
        let mut out = [0.0f32; 64];
        let noise: Vec<f32> = (0..32).map(|i| ((i * 37 % 19) as f32) - 9.0).collect();
        for _ in 0..12 {
            analysis.process_slot(&noise, &mut bands);
            let mut wide = [Complex32::default(); 64];
            wide[..32].copy_from_slice(&bands);
            synthesis.process_slot(&wide, &mut out);
        }
        analysis.reset();
        synthesis.reset();

        let zeros = [0.0f32; 32];
        for _ in 0..10 {
            analysis.process_slot(&zeros, &mut bands);
            let wide = [Complex32::default(); 64];
            synthesis.process_slot(&wide, &mut out);
        }
        assert!(bands.iter().all(|c| c.re == 0.0 && c.im == 0.0));
        assert!(out.iter().all(|&v| v == 0.0));
    }
}
