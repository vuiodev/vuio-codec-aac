//! Inverse MDCT via a quarter-length complex FFT.
//!
//! The IMDCT maps `n` spectral lines to `2n` time samples:
//!
//! ```text
//! x[i] = sum(k = 0..n) X[k] * cos(pi/n * (i + 1/2 + n/2) * (k + 1/2))
//! ```
//!
//! Evaluating that directly costs `O(n^2)`. This module uses the standard
//! decomposition into an `n/2`-point complex FFT wrapped in a pre-rotation and a
//! post-rotation, bringing the cost to `O(n log n)` and removing the need for a
//! large cosine matrix.
//!
//! The output has two symmetries — the first `n` samples are antisymmetric about
//! their midpoint and the last `n` are symmetric about theirs — so the `n` real
//! values the FFT produces expand to all `2n` outputs by index mapping alone.
//! [`ImdctContext::imdct`] therefore performs no redundant arithmetic.

use crate::dsp::fft::{Complex32, FftContext};
use std::f64::consts::PI;

/// Precomputed rotation factors and FFT plan for one IMDCT size.
#[derive(Debug, Clone)]
pub struct ImdctContext {
    /// Number of spectral lines.
    pub n: usize,
    /// FFT length, `n / 2`.
    m: usize,
    fft: FftContext,
    /// `exp(i * 2*pi*(k + 1/8) / (2n))` for `k` in `0..m`, premultiplied by the
    /// transform's normalization so the scaling costs nothing at runtime.
    pre_twiddle: Vec<Complex32>,
    /// The same rotation without the normalization, for the post-rotation.
    post_twiddle: Vec<Complex32>,
}

impl ImdctContext {
    /// Build a context for `n` spectral lines with the AAC synthesis normalization.
    ///
    /// ISO/IEC 14496-3 clause 4.6.11.2.2 defines the inverse transform with a factor
    /// of `2/N` where `N` is the window length, which is twice the spectral line
    /// count -- so `1/n` here. The matching forward transform carries a factor of 2,
    /// which is what `filterbank::tests::long_window_overlap_add_reconstructs` uses.
    ///
    /// `n` must be a power of two of at least 8.
    pub fn new(n: usize) -> Self {
        Self::with_scale(n, 1.0 / n as f32)
    }

    /// Build a context with an explicit output scale.
    ///
    /// [`Self::new`] applies the normalization the standard specifies; this exists
    /// for callers that want the unnormalized transform, and for tests that compare
    /// against the raw definition.
    pub fn with_scale(n: usize, scale: f32) -> Self {
        assert!(n >= 8 && n.is_power_of_two(), "IMDCT size must be a power of two >= 8");
        let m = n / 2;
        let two_n = 2 * n;

        // Computed in f64 and rounded once: at n = 1024 an f32 angle would lose
        // several bits of the rotation phase.
        let angles: Vec<(f32, f32)> = (0..m)
            .map(|k| {
                let a = 2.0 * PI * (k as f64 + 0.125) / two_n as f64;
                (a.cos() as f32, a.sin() as f32)
            })
            .collect();

        // Folding the scale into the pre-rotation applies it exactly once.
        let pre_twiddle =
            angles.iter().map(|&(c, s)| Complex32::new(c * scale, s * scale)).collect();
        let post_twiddle = angles.iter().map(|&(c, s)| Complex32::new(c, s)).collect();

        Self { n, m, fft: FftContext::new(m), pre_twiddle, post_twiddle }
    }

    /// Transform `n` spectral lines into `2n` time samples.
    ///
    /// `scratch` must hold at least `n` complex values: the first half receives the
    /// pre-rotated input and the second half the FFT output.
    pub fn imdct(&self, spectral: &[f32], out: &mut [f32], scratch: &mut [Complex32]) {
        let n = self.n;
        let m = self.m;
        let q = n / 4;
        debug_assert_eq!(spectral.len(), n);
        debug_assert_eq!(out.len(), 2 * n);
        debug_assert!(scratch.len() >= 2 * m);

        let (pre, z) = scratch[..2 * m].split_at_mut(m);

        // Pre-rotation: pair line 2k with line n-1-2k and rotate.
        for k in 0..m {
            let a = spectral[2 * k];
            let b = spectral[n - 1 - 2 * k];
            let w = self.pre_twiddle[k];
            pre[k] = Complex32::new(a * w.re + b * w.im, b * w.re - a * w.im);
        }

        self.fft.forward_into(pre, z);

        // Post-rotation, in place.
        for k in 0..m {
            let v = z[k];
            let w = self.post_twiddle[k];
            z[k] = Complex32::new(v.re * w.re + v.im * w.im, v.im * w.re - v.re * w.im);
        }

        // Scatter the m complex results across all 2n outputs. Each value appears
        // twice, once in each half, which is exactly the pair of symmetries the
        // IMDCT output obeys.
        let (h0, rest) = out.split_at_mut(m);
        let (h1, rest) = rest.split_at_mut(m);
        let (h2, h3) = rest.split_at_mut(m);

        for t in 0..q {
            let lo = z[q - 1 - t];
            let hi = z[q + t];
            let head = z[t];
            let tail = z[m - 1 - t];

            h0[2 * t] = hi.re;
            h0[2 * t + 1] = -lo.im;

            h1[2 * t] = head.im;
            h1[2 * t + 1] = -tail.re;

            h2[2 * t] = hi.im;
            h2[2 * t + 1] = -lo.re;

            h3[2 * t] = -head.re;
            h3[2 * t + 1] = tail.im;
        }
    }
}

/// Brute-force IMDCT straight from the definition.
///
/// Quadratic in `n`, so only for tests and for validating [`ImdctContext`].
pub fn imdct_reference(spectral: &[f32], out: &mut [f64]) {
    let n = spectral.len();
    for (i, o) in out.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        let base = PI / n as f64 * (i as f64 + 0.5 + n as f64 / 2.0);
        for (k, &x) in spectral.iter().enumerate() {
            sum += x as f64 * (base * (k as f64 + 0.5)).cos();
        }
        *o = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_rel_error(n: usize, seed: u32) -> f64 {
        let spectral: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i as f32 * 0.017 + seed as f32).sin() * 3.0
                    + (i as f32 * 0.11).cos()
                    + (i as f32 * 1.7).sin() * 0.25;
                x * 100.0
            })
            .collect();

        let mut want = vec![0.0f64; 2 * n];
        imdct_reference(&spectral, &mut want);

        let ctx = ImdctContext::with_scale(n, 1.0);
        let mut got = vec![0.0f32; 2 * n];
        let mut scratch = vec![Complex32::default(); n];
        ctx.imdct(&spectral, &mut got, &mut scratch);

        let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
        got.iter()
            .zip(want.iter())
            .map(|(&g, &w)| (g as f64 - w).abs() / peak)
            .fold(0.0f64, f64::max)
    }

    /// The fast transform must agree with the definition at every AAC size.
    #[test]
    fn matches_the_definition_at_every_size() {
        for &n in &[8usize, 16, 32, 64, 128, 256, 512, 1024] {
            let err = max_rel_error(n, 0);
            assert!(err < 2e-5, "n={n}: relative error {err:.3e}");
        }
    }

    /// Independent inputs must not alias into each other through the scratch buffer.
    #[test]
    fn is_stable_across_inputs() {
        for seed in 0..8 {
            let err = max_rel_error(1024, seed);
            assert!(err < 2e-5, "seed {seed}: relative error {err:.3e}");
        }
    }

    /// The first half is antisymmetric and the second half symmetric; the index
    /// mapping is built on those identities, so they must hold exactly.
    #[test]
    fn output_symmetries_hold() {
        let n = 256;
        let spectral: Vec<f32> = (0..n).map(|i| (i * 37 % 101) as f32 - 50.0).collect();
        let ctx = ImdctContext::with_scale(n, 1.0);
        let mut out = vec![0.0f32; 2 * n];
        let mut scratch = vec![Complex32::default(); n];
        ctx.imdct(&spectral, &mut out, &mut scratch);

        for j in 0..n {
            assert_eq!(out[n - 1 - j], -out[j], "antisymmetry broken at {j}");
        }
        for j in 0..n {
            assert_eq!(out[2 * n - 1 - j], out[n + j], "symmetry broken at {j}");
        }
    }

    /// A single spectral line must produce the corresponding pure cosine.
    #[test]
    fn single_line_produces_a_cosine() {
        let n = 128;
        for &k in &[0usize, 1, 7, 63, 127] {
            let mut spectral = vec![0.0f32; n];
            spectral[k] = 1.0;

            let ctx = ImdctContext::with_scale(n, 1.0);
            let mut got = vec![0.0f32; 2 * n];
            let mut scratch = vec![Complex32::default(); n];
            ctx.imdct(&spectral, &mut got, &mut scratch);

            for (i, &g) in got.iter().enumerate() {
                let want = (PI / n as f64 * (i as f64 + 0.5 + n as f64 / 2.0) * (k as f64 + 0.5))
                    .cos();
                assert!((g as f64 - want).abs() < 1e-4, "k={k} i={i}: {g} vs {want}");
            }
        }
    }

    /// The transform is linear, so scaling the input scales the output.
    #[test]
    fn transform_is_linear() {
        let n = 256;
        let base: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).sin()).collect();
        let scaled: Vec<f32> = base.iter().map(|v| v * -3.5).collect();

        let ctx = ImdctContext::with_scale(n, 1.0);
        let mut scratch = vec![Complex32::default(); n];
        let mut a = vec![0.0f32; 2 * n];
        let mut b = vec![0.0f32; 2 * n];
        ctx.imdct(&base, &mut a, &mut scratch);
        ctx.imdct(&scaled, &mut b, &mut scratch);

        let peak = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12);
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((y - x * -3.5).abs() / peak < 1e-5, "line {i}: {y} vs {}", x * -3.5);
        }
    }
}
