//! Look-ahead peak limiter, the decoder's final output-stage safety net.
//!
//! Ported from `c/libxaac/decoder/ixheaacd_peak_limiter.c`
//! (`ixheaacd_peak_limiter_process_float`; the fixed-point variant and its
//! separate `ixheaacd_scale_adjust` pre-scaling step are the fixed-point twin
//! of this float path and are [`N/A`](../../../text/plan.txt) per this port's
//! float-only design).
//!
//! # What a look-ahead limiter is and why it needs delay
//!
//! A simple limiter that reacts only to samples it has already seen cannot
//! catch a sharp transient before it clips: by the time the gain has ramped
//! down, the peak is already through. This design instead *delays the audio*
//! by the limiter's attack time and drives the gain from a small window of
//! *upcoming* samples (tracked via [`PeakLimiter::max_buf`]), so the gain
//! reduction is already in place by the time the loud sample reaches the
//! output. The `attack_time_samples`-deep [`PeakLimiter::delayed_input`] ring
//! buffer is that delay line.
//!
//! # The two time constants
//!
//! Attack and release are not symmetric, and deliberately so:
//! * **Attack** (5 ms, [`DEFAULT_ATTACK_TIME_MS`]) sets both the look-ahead
//!   depth and how fast the gain is allowed to *fall* — fast enough to catch a
//!   transient within its look-ahead window.
//! * **Release** (50 ms, [`DEFAULT_RELEASE_TIME_MS`]) sets how fast the gain
//!   *recovers* afterwards — slow, because recovering instantly would pump the
//!   gain up and down audibly on a run of transients; recovering slowly lets
//!   one gain reduction cover a whole loud passage smoothly.
//!
//! Both are converted from a time constant to a per-sample multiplier once, in
//! [`PeakLimiter::new`], via `0.1^(1/(n+1))` — this is the standard way to turn
//! "gain closes to within 10% of its target in `n` samples" (a perceptual
//! attack/release *time*) into the single-pole recursive filter's coefficient.

/// Attack time in milliseconds (`DEFAULT_ATTACK_TIME_MS`): both the
/// look-ahead depth and the fastest the gain may fall.
pub const DEFAULT_ATTACK_TIME_MS: f32 = 5.0;
/// Release time in milliseconds (`DEFAULT_RELEASE_TIME_MS`): how slowly the
/// gain recovers once the loud passage has passed.
pub const DEFAULT_RELEASE_TIME_MS: f32 = 50.0;
/// The ceiling no output sample may exceed (`PEAK_LIM_THR_FLOAT`), on the same
/// `±32768`-scaled convention this crate's PCM buffers already use.
pub const THRESHOLD: f32 = 29203.6;

/// A look-ahead peak limiter across `num_channels` channels, sharing one
/// envelope (the loudest channel at each instant drives the gain applied to
/// all channels, so a limited stereo signal does not shift image balance).
pub struct PeakLimiter {
    num_channels: usize,
    sample_rate_hz: u32,
    attack_time_samples: usize,
    attack_constant: f32,
    release_constant: f32,
    /// Sliding window of per-sample peak-across-channels magnitude, one entry
    /// per look-ahead sample; `max_buf[max_idx]` is the window's current max.
    max_buf: Vec<f32>,
    max_idx: usize,
    /// Where the next incoming peak overwrites `max_buf` (wraps at
    /// `attack_time_samples`).
    write_ptr: usize,
    /// The `attack_time_samples`-deep delay line, channel-interleaved.
    delayed_input: Vec<f32>,
    delayed_input_index: usize,
    gain_modified: f32,
    /// `f64` because the recursive smoothing accumulates many samples per
    /// frame and the reference keeps it in double precision for exactly that
    /// reason.
    pre_smoothed_gain: f64,
    /// The last frame's smallest applied gain -- exposed for callers that want
    /// to know how hard the limiter worked, matching `min_gain` in the
    /// reference's state struct.
    pub last_min_gain: f32,
}

impl PeakLimiter {
    /// Build a limiter for `num_channels` channels at `sample_rate_hz`. The
    /// look-ahead delay this introduces is [`PeakLimiter::delay_samples`].
    pub fn new(num_channels: usize, sample_rate_hz: u32) -> Self {
        let attack_time_samples =
            ((DEFAULT_ATTACK_TIME_MS * sample_rate_hz as f32 / 1000.0) as usize).max(1);
        let release_samples = DEFAULT_RELEASE_TIME_MS * sample_rate_hz as f32 / 1000.0;
        Self {
            num_channels,
            sample_rate_hz,
            attack_time_samples,
            attack_constant: 0.1f32.powf(1.0 / (attack_time_samples as f32 + 1.0)),
            release_constant: 0.1f32.powf(1.0 / (release_samples + 1.0)),
            max_buf: vec![0.0; attack_time_samples],
            max_idx: 0,
            write_ptr: 0,
            delayed_input: vec![0.0; attack_time_samples * num_channels],
            delayed_input_index: 0,
            gain_modified: 1.0,
            pre_smoothed_gain: 1.0,
            last_min_gain: 1.0,
        }
    }

    /// Samples of latency this limiter adds to the signal path.
    pub fn delay_samples(&self) -> usize {
        self.attack_time_samples
    }

    /// The channel count this instance was built for.
    pub fn channels(&self) -> usize {
        self.num_channels
    }

    /// The sample rate this instance was built for.
    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    /// Limit one frame in place. `samples[ch]` must all have the same length
    /// and there must be exactly [`PeakLimiter::num_channels`] of them.
    ///
    /// Because of the look-ahead delay, sample `i` written to the output on
    /// this call is the audio that was *input* `attack_time_samples` calls'
    /// worth of samples ago -- the first `delay_samples()` samples of the very
    /// first call are always silence, and the true tail of the stream is
    /// still sitting in the delay line after the last call (a caller wanting
    /// every sample out needs to flush with `attack_time_samples` samples of
    /// trailing silence).
    pub fn process(&mut self, samples: &mut [&mut [f32]]) {
        assert_eq!(samples.len(), self.num_channels, "one slice per configured channel");
        let frame_len = samples.first().map_or(0, |c| c.len());
        debug_assert!(samples.iter().all(|c| c.len() == frame_len));

        let mut min_gain = 1.0f32;
        for i in 0..frame_len {
            let peak = (0..self.num_channels).fold(0.0f32, |m, ch| m.max(samples[ch][i].abs()));
            self.max_buf[self.write_ptr] = peak;

            if self.max_idx == self.write_ptr {
                // The sample the current max came from is about to be
                // overwritten: the max may or may not still be in the window,
                // so this is the one case that needs a full rescan.
                self.max_idx = 0;
                for j in 1..self.attack_time_samples {
                    if self.max_buf[j] > self.max_buf[self.max_idx] {
                        self.max_idx = j;
                    }
                }
            } else if peak >= self.max_buf[self.max_idx] {
                self.max_idx = self.write_ptr;
            }

            self.write_ptr += 1;
            if self.write_ptr == self.attack_time_samples {
                self.write_ptr = 0;
            }

            let maximum = self.max_buf[self.max_idx];
            let gain = if maximum > THRESHOLD { THRESHOLD / maximum } else { 1.0 };

            // Two-stage smoothing: `gain_modified` reacts fast (it is allowed
            // to fall below the previous smoothed gain by up to 10% per
            // sample, the `* 1.111...` being 1/0.9 so a 10% closer target
            // still counts as "caught up"), then `pre_smoothed_gain` applies
            // the actual attack/release time constant on top of that.
            if gain < self.pre_smoothed_gain as f32 {
                self.gain_modified = self
                    .gain_modified
                    .min((gain - 0.1 * self.pre_smoothed_gain as f32) * (1.0 / 0.9));
            } else {
                self.gain_modified = gain;
            }

            if self.gain_modified < self.pre_smoothed_gain as f32 {
                self.pre_smoothed_gain = self.attack_constant as f64
                    * (self.pre_smoothed_gain - self.gain_modified as f64)
                    + self.gain_modified as f64;
                self.pre_smoothed_gain = self.pre_smoothed_gain.max(gain as f64);
            } else {
                self.pre_smoothed_gain = self.release_constant as f64
                    * (self.pre_smoothed_gain - self.gain_modified as f64)
                    + self.gain_modified as f64;
            }

            let applied_gain = self.pre_smoothed_gain as f32;
            for (ch, channel) in samples.iter_mut().enumerate() {
                let slot = &mut self.delayed_input[self.delayed_input_index * self.num_channels + ch];
                let delayed = *slot;
                *slot = channel[i];
                channel[i] = (delayed * applied_gain).clamp(-THRESHOLD, THRESHOLD);
            }

            self.delayed_input_index += 1;
            if self.delayed_input_index >= self.attack_time_samples {
                self.delayed_input_index = 0;
            }

            min_gain = min_gain.min(applied_gain);
        }
        self.last_min_gain = min_gain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limiter(channels: usize) -> PeakLimiter {
        PeakLimiter::new(channels, 44_100)
    }

    /// Silence must pass straight through, delayed but otherwise untouched,
    /// and must never engage the limiter (min gain stays 1.0).
    #[test]
    fn silence_passes_through_unchanged_after_the_look_ahead_delay() {
        let mut lim = limiter(1);
        let delay = lim.delay_samples();
        let mut buf = vec![0.0f32; delay * 3];
        let mut ch: Vec<&mut [f32]> = vec![&mut buf];
        lim.process(&mut ch);
        assert!(buf.iter().all(|x| *x == 0.0));
        assert_eq!(lim.last_min_gain, 1.0);
    }

    /// The core safety guarantee: however loud the input, no output sample
    /// may exceed the threshold. This is checked as a black-box property
    /// against many random loud signals, independent of matching the
    /// reference's exact internals.
    #[test]
    fn output_never_exceeds_the_threshold_for_arbitrarily_loud_input() {
        let mut lim = limiter(2);
        let n = 4096;
        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32 - 0.5) * 2.0
        };
        let mut left: Vec<f32> = (0..n).map(|_| next() * 200_000.0).collect();
        let mut right: Vec<f32> = (0..n).map(|_| next() * 200_000.0).collect();
        let mut ch: Vec<&mut [f32]> = vec![&mut left, &mut right];
        lim.process(&mut ch);

        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(l.abs() <= THRESHOLD + 1e-3, "left[{i}] = {l} exceeds threshold");
            assert!(r.abs() <= THRESHOLD + 1e-3, "right[{i}] = {r} exceeds threshold");
        }
        assert!(lim.last_min_gain < 1.0, "a loud signal must have engaged the limiter");
    }

    /// A quiet signal well under the threshold must be left at unity gain and
    /// pass through with only the look-ahead delay applied -- no gain
    /// reduction where none is needed.
    #[test]
    fn a_quiet_signal_is_untouched_besides_the_delay() {
        let mut lim = limiter(1);
        let delay = lim.delay_samples();
        let n = delay * 4;
        let input: Vec<f32> = (0..n).map(|i| 100.0 * ((i as f32) * 0.1).sin()).collect();
        let mut buf = input.clone();
        {
            let mut ch: Vec<&mut [f32]> = vec![&mut buf];
            lim.process(&mut ch);
        }
        assert_eq!(lim.last_min_gain, 1.0);
        // Past the initial look-ahead delay, output[i] must equal input[i-delay].
        for i in delay..n {
            assert!((buf[i] - input[i - delay]).abs() < 1e-2, "i={i}: {} vs {}", buf[i], input[i - delay]);
        }
    }

    /// A brute-force twin of the whole limiter that recomputes the windowed
    /// max from scratch every sample (`O(n*window)` instead of the real
    /// implementation's amortized `O(n)` lazy-rescan) must produce identical
    /// output. This isolates exactly the piece most likely to hide an
    /// off-by-one -- the incremental window-max bookkeeping -- from the
    /// smoothing math, which both implementations share.
    fn brute_force_limiter(window: usize, attack: f32, release: f32, input: &[f32]) -> Vec<f32> {
        let mut history = vec![0.0f32; input.len()];
        let mut delayed = vec![0.0f32; window];
        let mut delayed_index = 0usize;
        let mut gain_modified = 1.0f32;
        let mut pre_smoothed = 1.0f64;
        let mut out = vec![0.0f32; input.len()];

        for (i, &x) in input.iter().enumerate() {
            history[i] = x.abs();
            let lo = i.saturating_sub(window - 1);
            let maximum = history[lo..=i].iter().cloned().fold(0.0f32, f32::max);
            let gain = if maximum > THRESHOLD { THRESHOLD / maximum } else { 1.0 };

            if gain < pre_smoothed as f32 {
                gain_modified = gain_modified.min((gain - 0.1 * pre_smoothed as f32) * (1.0 / 0.9));
            } else {
                gain_modified = gain;
            }
            if gain_modified < pre_smoothed as f32 {
                pre_smoothed = attack as f64 * (pre_smoothed - gain_modified as f64) + gain_modified as f64;
                pre_smoothed = pre_smoothed.max(gain as f64);
            } else {
                pre_smoothed = release as f64 * (pre_smoothed - gain_modified as f64) + gain_modified as f64;
            }

            let applied = pre_smoothed as f32;
            let d = delayed[delayed_index];
            delayed[delayed_index] = x;
            out[i] = (d * applied).clamp(-THRESHOLD, THRESHOLD);
            delayed_index = (delayed_index + 1) % window;
        }
        out
    }

    #[test]
    fn the_lazy_rescan_max_tracker_matches_a_brute_force_twin() {
        let mut lim = limiter(1);
        let window = lim.delay_samples();
        let n = window * 5;
        let mut seed = 42u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed % 200_000) as f32 - 100_000.0
        };
        let input: Vec<f32> = (0..n).map(|_| next()).collect();

        let mut buf = input.clone();
        {
            let mut ch: Vec<&mut [f32]> = vec![&mut buf];
            lim.process(&mut ch);
        }

        let attack = 0.1f32.powf(1.0 / (window as f32 + 1.0));
        let release_samples = DEFAULT_RELEASE_TIME_MS * 44_100.0 / 1000.0;
        let release = 0.1f32.powf(1.0 / (release_samples + 1.0));
        let want = brute_force_limiter(window, attack, release, &input);

        for i in 0..n {
            assert!((buf[i] - want[i]).abs() < 1e-2, "i={i}: fast {} vs brute-force {}", buf[i], want[i]);
        }
    }
}
