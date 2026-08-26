//! AAC synthesis filterbank: IMDCT, windowing and overlap-add.
//!
//! Reconstructs time-domain samples from spectral coefficients. Each frame's IMDCT
//! produces `2n` samples that span two frame periods; the first half overlap-adds
//! with the tail the previous frame left behind, and the second half becomes the
//! tail for the next one. That overlap is what cancels the time-domain aliasing the
//! MDCT introduced (the Princen-Bradley condition).
//!
//! The four window sequences exist to switch between the long transform (good
//! frequency resolution) and eight short transforms (good time resolution) without
//! breaking that cancellation. A start window has a long rising edge and a short
//! falling edge; a stop window is its mirror. See ISO/IEC 14496-3 clause 4.6.11.

use crate::dsp::fft::Complex32;
use crate::dsp::imdct::ImdctContext;
use crate::dsp::window::{generate_kbd_window_f32, generate_sine_window_f32};
use crate::types::{WindowSequence, WindowShape};

/// KBD shape parameter for long windows, per the standard.
const KBD_ALPHA_LONG: f32 = 4.0;
/// KBD shape parameter for short windows.
const KBD_ALPHA_SHORT: f32 = 6.0;

/// The four window halves the filterbank selects between.
#[derive(Debug, Clone)]
struct WindowSet {
    /// Length `2n`, sine shape.
    long_sine: Vec<f32>,
    /// Length `2n`, KBD shape.
    long_kbd: Vec<f32>,
    /// Length `2n/8`, sine shape.
    short_sine: Vec<f32>,
    /// Length `2n/8`, KBD shape.
    short_kbd: Vec<f32>,
}

impl WindowSet {
    fn new(n: usize, short_n: usize) -> Self {
        Self {
            long_sine: generate_sine_window_f32(2 * n),
            long_kbd: generate_kbd_window_f32(2 * n, KBD_ALPHA_LONG),
            short_sine: generate_sine_window_f32(2 * short_n),
            short_kbd: generate_kbd_window_f32(2 * short_n, KBD_ALPHA_SHORT),
        }
    }

    #[inline]
    fn long(&self, shape: WindowShape) -> &[f32] {
        match shape {
            WindowShape::Kbd => &self.long_kbd,
            _ => &self.long_sine,
        }
    }

    #[inline]
    fn short(&self, shape: WindowShape) -> &[f32] {
        match shape {
            WindowShape::Kbd => &self.short_kbd,
            _ => &self.short_sine,
        }
    }
}

/// Build the analysis/synthesis window a frame uses, as a `2n` coefficient vector.
///
/// The decoder applies these implicitly while overlap-adding; materialising them is
/// what lets a test run the matching forward transform and check reconstruction.
pub fn frame_window(
    n: usize,
    sequence: WindowSequence,
    shape: WindowShape,
    prev_shape: WindowShape,
) -> Vec<f32> {
    let sn = n / 8;
    let flat = (n - sn) / 2;
    let set = WindowSet::new(n, sn);
    let rise_long = set.long(prev_shape);
    let rise_short = set.short(prev_shape);
    let fall_long = set.long(shape);
    let fall_short = set.short(shape);

    let mut w = vec![0.0f32; 2 * n];
    match sequence {
        WindowSequence::OnlyLongSequence => {
            w[..n].copy_from_slice(&rise_long[..n]);
            w[n..].copy_from_slice(&fall_long[n..]);
        }
        WindowSequence::LongStartSequence => {
            w[..n].copy_from_slice(&rise_long[..n]);
            w[n..n + flat].fill(1.0);
            w[n + flat..n + flat + sn].copy_from_slice(&fall_short[sn..]);
        }
        WindowSequence::LongStopSequence => {
            w[flat..flat + sn].copy_from_slice(&rise_short[..sn]);
            w[flat + sn..n].fill(1.0);
            w[n..].copy_from_slice(&fall_long[n..]);
        }
        WindowSequence::EightShortSequence => {
            // Not a single window; the eight sub-windows are applied individually.
        }
    }
    w
}

/// The rising and falling halves of one short sub-window.
pub fn short_window(n: usize, shape: WindowShape, prev_shape: WindowShape, index: usize) -> Vec<f32> {
    let sn = n / 8;
    let set = WindowSet::new(n, sn);
    let rise = if index == 0 { set.short(prev_shape) } else { set.short(shape) };
    let fall = set.short(shape);
    let mut w = vec![0.0f32; 2 * sn];
    w[..sn].copy_from_slice(&rise[..sn]);
    w[sn..].copy_from_slice(&fall[sn..]);
    w
}

/// Offset of short sub-window `index` inside the `2n` frame block.
pub const fn short_window_offset(n: usize, index: usize) -> usize {
    let sn = n / 8;
    (n - sn) / 2 + index * sn
}

/// Per-frame synthesis filterbank for one transform size.
#[derive(Debug, Clone)]
pub struct Filterbank {
    /// Spectral lines per long frame.
    pub n: usize,
    /// Spectral lines per short window, `n / 8`.
    pub short_n: usize,
    imdct_long: ImdctContext,
    imdct_short: ImdctContext,
    windows: WindowSet,
    /// `2n` IMDCT output / windowed accumulator.
    time: Vec<f32>,
    /// Scratch for one short IMDCT.
    short_time: Vec<f32>,
    fft_scratch: Vec<Complex32>,
}

impl Filterbank {
    /// Build a filterbank for `n` spectral lines per long frame.
    pub fn new(n: usize) -> Self {
        let short_n = n / 8;
        Self {
            n,
            short_n,
            imdct_long: ImdctContext::new(n),
            imdct_short: ImdctContext::new(short_n),
            windows: WindowSet::new(n, short_n),
            time: vec![0.0; 2 * n],
            short_time: vec![0.0; 2 * short_n],
            fft_scratch: vec![Complex32::default(); n],
        }
    }

    /// Transform one channel's spectrum and overlap-add into `out`.
    ///
    /// `spectral` holds `n` lines, laid out per-window for short sequences.
    /// `overlap` carries `n` samples of tail across frames and is updated in place.
    /// `prev_shape` is the window shape the previous frame ended with, which decides
    /// the shape of this frame's rising edge.
    pub fn synthesize(
        &mut self,
        spectral: &[f32],
        sequence: WindowSequence,
        shape: WindowShape,
        prev_shape: WindowShape,
        overlap: &mut [f32],
        out: &mut [f32],
    ) {
        let n = self.n;

        match sequence {
            WindowSequence::EightShortSequence => {
                self.synthesize_short(spectral, shape, prev_shape);
            }
            _ => {
                self.imdct_long.imdct(&spectral[..n], &mut self.time, &mut self.fft_scratch);
                self.apply_long_window(sequence, shape, prev_shape);
            }
        }

        // First half overlaps with the previous frame's tail; second half is saved.
        for i in 0..n {
            out[i] = self.time[i] + overlap[i];
            overlap[i] = self.time[n + i];
        }
    }

    /// Window a long/start/stop transform in place over `self.time`.
    fn apply_long_window(
        &mut self,
        sequence: WindowSequence,
        shape: WindowShape,
        prev_shape: WindowShape,
    ) {
        let n = self.n;
        let sn = self.short_n;
        // A short window's flat region is centred in the long frame: the transition
        // occupies `sn` samples starting `(n - sn) / 2` in from the edge.
        let flat = (n - sn) / 2;

        let rise_long = self.windows.long(prev_shape);
        let rise_short = self.windows.short(prev_shape);
        let fall_long = self.windows.long(shape);
        let fall_short = self.windows.short(shape);

        match sequence {
            WindowSequence::OnlyLongSequence => {
                for i in 0..n {
                    self.time[i] *= rise_long[i];
                    self.time[n + i] *= fall_long[n + i];
                }
            }
            WindowSequence::LongStartSequence => {
                // Long rising edge, then flat, then a short falling edge, then zero.
                for i in 0..n {
                    self.time[i] *= rise_long[i];
                }
                for i in 0..flat {
                    // Flat region: coefficient is 1, nothing to do.
                    let _ = i;
                }
                for i in 0..sn {
                    self.time[n + flat + i] *= fall_short[sn + i];
                }
                self.time[n + flat + sn..].fill(0.0);
            }
            WindowSequence::LongStopSequence => {
                // Mirror of the start window: zero, short rising edge, flat, then a
                // long falling edge.
                self.time[..flat].fill(0.0);
                for i in 0..sn {
                    self.time[flat + i] *= rise_short[i];
                }
                for i in 0..n {
                    self.time[n + i] *= fall_long[n + i];
                }
            }
            WindowSequence::EightShortSequence => unreachable!("handled by synthesize_short"),
        }
    }

    /// Transform and overlap the eight short windows into `self.time`.
    fn synthesize_short(&mut self, spectral: &[f32], shape: WindowShape, prev_shape: WindowShape) {
        let n = self.n;
        let sn = self.short_n;
        // The eight short transforms sit centred inside the long frame.
        let base = (n - sn) / 2;

        self.time.fill(0.0);

        for w in 0..8 {
            let lo = w * sn;
            self.imdct_short.imdct(
                &spectral[lo..lo + sn],
                &mut self.short_time,
                &mut self.fft_scratch,
            );

            // Only the very first short window's rising edge continues the previous
            // frame's shape; the rest are all current-shape.
            let rise = if w == 0 {
                self.windows.short(prev_shape)
            } else {
                self.windows.short(shape)
            };
            let fall = self.windows.short(shape);

            let start = base + w * sn;
            for i in 0..sn {
                self.time[start + i] += self.short_time[i] * rise[i];
                self.time[start + sn + i] += self.short_time[sn + i] * fall[sn + i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both window shapes must satisfy the Princen-Bradley condition, without which
    /// overlap-add cannot cancel the time-domain aliasing.
    #[test]
    fn windows_satisfy_princen_bradley() {
        for &len in &[256usize, 2048] {
            for win in [
                generate_sine_window_f32(len),
                generate_kbd_window_f32(len, if len == 256 { KBD_ALPHA_SHORT } else { KBD_ALPHA_LONG }),
            ] {
                let half = len / 2;
                for i in 0..half {
                    let s = win[i] * win[i] + win[half + i] * win[half + i];
                    assert!((s - 1.0).abs() < 1e-4, "len {len} index {i}: {s}");
                }
            }
        }
    }

    /// A constant signal pushed through MDCT and back must reconstruct exactly once
    /// the overlap-add has had a frame to prime. This is the end-to-end check that
    /// the transform, windows and overlap all agree.
    #[test]
    fn long_window_overlap_add_reconstructs() {
        let n = 256;
        let mut fb = Filterbank::new(n);

        // Build a signal and forward-MDCT it frame by frame with the same window.
        let total = n * 6;
        let signal: Vec<f32> = (0..total + 2 * n)
            .map(|i| ((i as f32) * 0.021).sin() * 1000.0 + ((i as f32) * 0.0037).cos() * 400.0)
            .collect();
        let window = generate_sine_window_f32(2 * n);

        let mut overlap = vec![0.0f32; n];
        let mut recon = vec![0.0f32; total];

        for f in 0..total / n {
            let start = f * n;
            // Forward MDCT of the windowed 2n-sample block, from the definition.
            let mut spec = vec![0.0f32; n];
            for (k, s) in spec.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for i in 0..2 * n {
                    let a = std::f64::consts::PI / n as f64
                        * (i as f64 + 0.5 + n as f64 / 2.0)
                        * (k as f64 + 0.5);
                    acc += (signal[start + i] * window[i]) as f64 * a.cos();
                }
                // The standard's forward transform carries a factor of 2.
                *s = (acc * 2.0) as f32;
            }

            let mut out = vec![0.0f32; n];
            fb.synthesize(
                &spec,
                WindowSequence::OnlyLongSequence,
                WindowShape::Sine,
                WindowShape::Sine,
                &mut overlap,
                &mut out,
            );
            recon[start..start + n].copy_from_slice(&out);
        }

        // The first frame has no prior overlap, so start comparing from frame 1.
        for i in n..total {
            let want = signal[i];
            let got = recon[i];
            assert!(
                (got - want).abs() < 1.0,
                "sample {i}: reconstructed {got}, expected {want}"
            );
        }
    }

    /// Short-window synthesis must place all eight transforms inside the frame and
    /// leave the guard regions untouched.
    #[test]
    fn short_windows_stay_inside_the_frame() {
        let n = 1024;
        let mut fb = Filterbank::new(n);
        let spectral: Vec<f32> = (0..n).map(|i| ((i % 128) as f32) - 64.0).collect();
        let mut overlap = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];

        fb.synthesize(
            &spectral,
            WindowSequence::EightShortSequence,
            WindowShape::Sine,
            WindowShape::Sine,
            &mut overlap,
            &mut out,
        );

        let sn = n / 8;
        let base = (n - sn) / 2;
        // Nothing may be written before the first short window starts.
        for i in 0..base {
            assert_eq!(out[i], 0.0, "leading guard sample {i} was written");
        }
        assert!(out.iter().any(|&v| v != 0.0), "short synthesis produced silence");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// A start window must decay to zero at its trailing edge, and a stop window
    /// must be zero at its leading edge.
    #[test]
    fn transition_windows_have_zero_guard_regions() {
        let n = 256;
        let sn = n / 8;
        let flat = (n - sn) / 2;
        let mut fb = Filterbank::new(n);
        let spectral: Vec<f32> = (0..n).map(|i| (i as f32 % 17.0) - 8.0).collect();
        let mut overlap = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];

        // A start window's tail is the overlap handed to the next frame.
        fb.synthesize(
            &spectral,
            WindowSequence::LongStartSequence,
            WindowShape::Sine,
            WindowShape::Sine,
            &mut overlap,
            &mut out,
        );
        for i in flat + sn..n {
            assert_eq!(overlap[i], 0.0, "start window tail {i} is not zero");
        }

        // A stop window's head must be zero.
        let mut overlap2 = vec![0.0f32; n];
        let mut out2 = vec![0.0f32; n];
        fb.synthesize(
            &spectral,
            WindowSequence::LongStopSequence,
            WindowShape::Sine,
            WindowShape::Sine,
            &mut overlap2,
            &mut out2,
        );
        for i in 0..flat {
            assert_eq!(out2[i], 0.0, "stop window head {i} is not zero");
        }
    }

    /// Silence in must give silence out, for every sequence.
    #[test]
    fn silence_produces_silence() {
        let n = 1024;
        let mut fb = Filterbank::new(n);
        for seq in [
            WindowSequence::OnlyLongSequence,
            WindowSequence::LongStartSequence,
            WindowSequence::EightShortSequence,
            WindowSequence::LongStopSequence,
        ] {
            let mut overlap = vec![0.0f32; n];
            let mut out = vec![1.0f32; n];
            fb.synthesize(
                &vec![0.0; n],
                seq,
                WindowShape::Sine,
                WindowShape::Sine,
                &mut overlap,
                &mut out,
            );
            assert!(out.iter().all(|&v| v == 0.0), "{seq:?} leaked nonzero output");
            assert!(overlap.iter().all(|&v| v == 0.0), "{seq:?} leaked nonzero overlap");
        }
    }
}
