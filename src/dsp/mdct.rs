//! Forward MDCT.
//!
//! Maps `2n` windowed time samples to `n` spectral coefficients:
//!
//! ```text
//! X[k] = 2 * sum(i = 0..2n) x[i] * cos(pi/n * (i + 1/2 + n/2) * (k + 1/2))
//! ```
//!
//! The factor of two is the analysis half of the normalization whose synthesis half
//! is the `1/n` in [`crate::dsp::imdct`]; together they make windowed overlap-add
//! reconstruct the original signal.
//!
//! Computed through a `2n`-point complex FFT: pre-rotate the real input, transform,
//! then take a rotated real part. The inverse transform uses a smaller `n/2`-point
//! FFT; the forward direction is the encoder's cost and runs once per frame, so the
//! simpler decomposition is kept for clarity.

use crate::dsp::fft::{Complex32, FftContext};
use std::f64::consts::PI;

/// Precomputed rotations and FFT plan for one forward transform size.
#[derive(Debug, Clone)]
pub struct MdctContext {
    /// Number of spectral coefficients produced.
    pub n: usize,
    fft: FftContext,
    /// `exp(-i*pi*m/(2n))` for `m` in `0..2n`.
    pre: Vec<Complex32>,
    /// `exp(i*theta_k)` with `theta_k = pi*(k + 1/2)*(n + 1)/(2n)`.
    post: Vec<Complex32>,
    /// Output scale.
    scale: f32,
}

impl MdctContext {
    /// Build a context producing `n` coefficients, with the standard analysis scale.
    pub fn new(n: usize) -> Self {
        Self::with_scale(n, 2.0)
    }

    /// Build a context with an explicit output scale.
    pub fn with_scale(n: usize, scale: f32) -> Self {
        assert!(n >= 4 && n.is_power_of_two(), "MDCT size must be a power of two >= 4");
        let two_n = 2 * n;

        let pre = (0..two_n)
            .map(|m| {
                let a = -PI * m as f64 / two_n as f64;
                Complex32::new(a.cos() as f32, a.sin() as f32)
            })
            .collect();

        let post = (0..n)
            .map(|k| {
                let theta = PI * (k as f64 + 0.5) * (n as f64 + 1.0) / two_n as f64;
                Complex32::new(theta.cos() as f32, theta.sin() as f32)
            })
            .collect();

        Self { n, fft: FftContext::new(two_n), pre, post, scale }
    }

    /// Transform `2n` windowed samples into `n` coefficients.
    ///
    /// `scratch` must hold at least `4n` complex values: `2n` for the rotated input
    /// and `2n` for the FFT output.
    pub fn forward(&self, time: &[f32], spec: &mut [f32], scratch: &mut [Complex32]) {
        let n = self.n;
        let two_n = 2 * n;
        assert_eq!(time.len(), two_n);
        assert_eq!(spec.len(), n);
        assert!(scratch.len() >= 2 * two_n);

        let (input, output) = scratch[..2 * two_n].split_at_mut(two_n);
        for m in 0..two_n {
            let w = self.pre[m];
            let x = time[m];
            input[m] = Complex32::new(x * w.re, x * w.im);
        }

        self.fft.forward_into(input, output);

        for k in 0..n {
            let w = self.post[k];
            let z = output[k];
            spec[k] = (z.re * w.re + z.im * w.im) * self.scale;
        }
    }

    /// Scratch length [`Self::forward`] requires.
    pub const fn scratch_len(&self) -> usize {
        4 * self.n
    }
}

/// Brute-force forward MDCT from the definition, for tests.
pub fn mdct_reference(time: &[f32], spec: &mut [f64], scale: f64) {
    let n = spec.len();
    for (k, s) in spec.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for (i, &x) in time.iter().enumerate() {
            let a = PI / n as f64 * (i as f64 + 0.5 + n as f64 / 2.0) * (k as f64 + 0.5);
            acc += x as f64 * a.cos();
        }
        *s = acc * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::imdct::ImdctContext;

    fn signal(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 + seed as f32 * 1.7;
                (t * 0.037).sin() * 10000.0 + (t * 0.13).cos() * 5000.0 + (t * 0.9).sin() * 700.0
            })
            .collect()
    }

    /// The fast transform must match the definition at every size the codec uses.
    #[test]
    fn matches_the_definition() {
        for &n in &[16usize, 64, 128, 256, 512, 1024] {
            let time = signal(2 * n, 1);
            let mut want = vec![0.0f64; n];
            mdct_reference(&time, &mut want, 2.0);

            let ctx = MdctContext::new(n);
            let mut got = vec![0.0f32; n];
            let mut scratch = vec![Complex32::default(); ctx.scratch_len()];
            ctx.forward(&time, &mut got, &mut scratch);

            let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-9);
            for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                let e = (*g as f64 - w).abs() / peak;
                assert!(e < 2e-5, "n={n} k={k}: {g} vs {w} (rel {e:.3e})");
            }
        }
    }

    /// Forward then inverse, with the matching normalizations, must satisfy the
    /// time-domain aliasing identity: the sum of two overlapped frames reconstructs.
    #[test]
    fn forward_and_inverse_normalizations_pair_up() {
        let n = 256;
        let fwd = MdctContext::new(n);
        let inv = ImdctContext::new(n);

        let time = signal(2 * n, 3);
        let mut spec = vec![0.0f32; n];
        let mut fscratch = vec![Complex32::default(); fwd.scratch_len()];
        fwd.forward(&time, &mut spec, &mut fscratch);

        let mut back = vec![0.0f32; 2 * n];
        let mut iscratch = vec![Complex32::default(); n];
        inv.imdct(&spec, &mut back, &mut iscratch);

        // A single unwindowed round trip does not return the input: the MDCT is
        // critically sampled, so half the information is folded away as time-domain
        // aliasing. What comes back in the first half is exactly the input minus its
        // own reflection. Overlap-add with the neighbouring frame is what cancels
        // that fold; `tests/filterbank_reconstruction.rs` covers the cancellation.
        for i in 0..n {
            let expected = time[i] - time[n - 1 - i];
            let tol = expected.abs() * 1e-3 + 1.0;
            assert!(
                (back[i] - expected).abs() < tol,
                "i={i}: {} vs {expected}",
                back[i]
            );
        }

        // The fold is antisymmetric about the midpoint of the first half.
        for i in 0..n / 2 {
            assert!(
                (back[i] + back[n - 1 - i]).abs() < back[i].abs() * 1e-3 + 1.0,
                "aliasing is not antisymmetric at {i}"
            );
        }
    }

    /// The transform is linear.
    #[test]
    fn is_linear() {
        let n = 128;
        let ctx = MdctContext::new(n);
        let mut scratch = vec![Complex32::default(); ctx.scratch_len()];

        let a = signal(2 * n, 1);
        let b = signal(2 * n, 9);
        let sum: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        let (mut fa, mut fb, mut fs) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        ctx.forward(&a, &mut fa, &mut scratch);
        ctx.forward(&b, &mut fb, &mut scratch);
        ctx.forward(&sum, &mut fs, &mut scratch);

        let peak = fs.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-9);
        for k in 0..n {
            assert!((fa[k] + fb[k] - fs[k]).abs() / peak < 1e-4, "k={k}");
        }
    }

    /// Silence in, silence out.
    #[test]
    fn silence_produces_silence() {
        let n = 64;
        let ctx = MdctContext::new(n);
        let mut scratch = vec![Complex32::default(); ctx.scratch_len()];
        let mut spec = vec![1.0f32; n];
        ctx.forward(&vec![0.0; 2 * n], &mut spec, &mut scratch);
        assert!(spec.iter().all(|v| v.abs() < 1e-6));
    }
}
