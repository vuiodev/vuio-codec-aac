//! Modified Discrete Cosine Transform (MDCT & IMDCT)
//!
//! Implements Forward MDCT via 2N-point FFT (O(N log N)) and Inverse MDCT (IMDCT)
//! with Time-Domain Aliasing Cancellation (TDAC), precomputed twiddle tables,
//! and zero-allocation overlap-add buffers.
//!
//! ## Forward MDCT Algorithm
//!
//! The forward MDCT is defined as:
//! $$X[k] = \frac{2}{N} \sum_{n=0}^{2N-1} x[n] \cos\left(\frac{\pi}{N}(n + \frac{1}{2} + \frac{N}{2})(k + \frac{1}{2})\right)$$
//!
//! This is decomposed into:
//! 1. **Pre-twiddle**: $z[m] = x[m] \cdot e^{-j\pi m/(2N)}$ for $m = 0..2N-1$
//! 2. **2N-point Forward FFT** on $z$
//! 3. **Post-twiddle**: $X[k] = \frac{2}{N} \text{Re}[e^{j\theta_k} \cdot \overline{Z[k]}]$
//!    where $\theta_k = \frac{\pi(k+0.5)(N+1)}{2N}$

use std::f64::consts::PI as PI64;
use crate::dsp::fft::{Complex32, FftContext};

/// Context holding precomputed FFT twiddles and transform engine for fast MDCT/IMDCT.
#[derive(Debug, Clone)]
pub struct MdctContext {
    /// Transform size N (number of spectral lines, e.g. 1024)
    pub n: usize,
    /// N/2-point FFT context (used for IMDCT)
    pub fft: FftContext,
    /// 2N-point FFT context (used for forward MDCT)
    pub fft_fwd: FftContext,
    /// Legacy IMDCT twiddles
    pub pre_twiddles: Vec<Complex32>,
    pub post_twiddles: Vec<Complex32>,
    /// Forward MDCT pre-twiddles: exp(-jπm/(2N)) for m = 0..2N-1
    pub fwd_pre_tw: Vec<Complex32>,
    /// Forward MDCT post-twiddles: (cos θ_k, sin θ_k) for k = 0..N-1
    pub fwd_post_tw: Vec<Complex32>,
    /// Precomputed 2N × N transposed cosine matrix for SIMD inverse MDCT dot-products
    pub cos_imdct: Vec<f32>,
}

impl MdctContext {
    /// Create MDCT context for transform size `n` (number of spectral lines, e.g. 1024 or 128).
    pub fn new(n: usize) -> Self {
        let n_half = n / 2;
        let fft = FftContext::new(n_half);
        let fft_fwd = FftContext::new(2 * n);

        // Legacy IMDCT pre/post twiddles
        let mut pre_twiddles = Vec::with_capacity(n_half);
        for k in 0..n_half {
            let angle = -PI64 * ((2.0 * k as f64 + 1.0) / (2.0 * n as f64));
            pre_twiddles.push(Complex32::new(angle.cos() as f32, angle.sin() as f32));
        }
        let mut post_twiddles = Vec::with_capacity(n_half);
        for k in 0..n_half {
            let angle = -PI64 * ((2.0 * k as f64 + 1.0) / (2.0 * n as f64));
            post_twiddles.push(Complex32::new(angle.cos() as f32, angle.sin() as f32));
        }

        // Forward MDCT pre-twiddles: exp(-jπm/(2N)) for m = 0..2N-1
        // Using f64 precision for twiddle precomputation
        let two_n = 2 * n;
        let mut fwd_pre_tw = Vec::with_capacity(two_n);
        for m in 0..two_n {
            let angle = -PI64 * m as f64 / (2.0 * n as f64);
            fwd_pre_tw.push(Complex32::new(angle.cos() as f32, angle.sin() as f32));
        }

        // Forward MDCT post-twiddles: (cos θ_k, sin θ_k) for k = 0..N-1
        // θ_k = π(k+0.5)(N+1)/(2N)
        let mut fwd_post_tw = Vec::with_capacity(n);
        for k in 0..n {
            let theta = PI64 * (k as f64 + 0.5) * (n as f64 + 1.0) / (2.0 * n as f64);
            fwd_post_tw.push(Complex32::new(theta.cos() as f32, theta.sin() as f32));
        }

        // Transposed Inverse cosine matrix: cos_imdct[i * N + k]
        let mut cos_imdct = vec![0.0f32; 2 * n * n];
        for i in 0..2 * n {
            let i_offset = i * n;
            let i_term = (i as f64 + 0.5 + (n as f64) / 2.0) * PI64 / (n as f64);
            for k in 0..n {
                let angle = (k as f64 + 0.5) * i_term;
                cos_imdct[i_offset + k] = angle.cos() as f32;
            }
        }

        Self {
            n,
            fft,
            fft_fwd,
            pre_twiddles,
            post_twiddles,
            fwd_pre_tw,
            fwd_post_tw,
            cos_imdct,
        }
    }

    /// FFT-based Forward MDCT: O(N log N) via 2N-point complex FFT.
    ///
    /// Transforms 2N windowed time-domain samples into N spectral coefficients.
    /// `scratch` must have length >= 2N.
    ///
    /// Derivation:
    /// ```text
    /// X[k] = (2/N) Re[exp(jθ_k) · conj(DFT₂ₙ[x[m]·exp(-jπm/(2N))]_k)]
    /// where θ_k = π(k+0.5)(N+1)/(2N)
    /// ```
    #[inline(always)]
    pub fn forward_mdct_fft(&self, time_in_2n: &[f32], spec_out_n: &mut [f32], scratch: &mut [Complex32]) {
        let n = self.n;
        let two_n = 2 * n;
        let scale = 2.0 / n as f32;

        assert_eq!(time_in_2n.len(), two_n);
        assert_eq!(spec_out_n.len(), n);
        assert!(scratch.len() >= two_n);

        // Step 1: Pre-twiddle — multiply real input by precomputed exp(-jπm/(2N))
        for m in 0..two_n {
            let tw = &self.fwd_pre_tw[m];
            let x = time_in_2n[m];
            scratch[m] = Complex32::new(x * tw.re, x * tw.im);
        }

        // Step 2: 2N-point forward FFT
        self.fft_fwd.forward(&mut scratch[..two_n]);

        // Step 3: Post-twiddle — extract MDCT coefficients
        // X[k] = (2/N) * [cos(θ_k) * Z[k].re + sin(θ_k) * Z[k].im]
        for k in 0..n {
            let tw = &self.fwd_post_tw[k];
            let z = &scratch[k];
            spec_out_n[k] = (z.re * tw.re + z.im * tw.im) * scale;
        }
    }

    /// 16-Way SIMD FMA-Unrolled Inverse MDCT: transforms N spectral coefficients into 2N time-domain samples.
    #[inline(always)]
    pub fn imdct(&self, input: &[f32], output: &mut [f32]) {
        let n = self.n;
        assert_eq!(input.len(), n, "Input must have N spectral coefficients");
        assert_eq!(output.len(), 2 * n, "Output must hold 2N time samples");

        let chunks_16 = n / 16;

        for (i, out) in output.iter_mut().enumerate().take(2 * n) {
            let row = &self.cos_imdct[i * n..(i + 1) * n];
            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut sum3 = 0.0f32;
            let mut sum4 = 0.0f32;
            let mut sum5 = 0.0f32;
            let mut sum6 = 0.0f32;
            let mut sum7 = 0.0f32;

            for c in 0..chunks_16 {
                let idx = c * 16;
                sum0 += input[idx] * row[idx] + input[idx + 8] * row[idx + 8];
                sum1 += input[idx + 1] * row[idx + 1] + input[idx + 9] * row[idx + 9];
                sum2 += input[idx + 2] * row[idx + 2] + input[idx + 10] * row[idx + 10];
                sum3 += input[idx + 3] * row[idx + 3] + input[idx + 11] * row[idx + 11];
                sum4 += input[idx + 4] * row[idx + 4] + input[idx + 12] * row[idx + 12];
                sum5 += input[idx + 5] * row[idx + 5] + input[idx + 13] * row[idx + 13];
                sum6 += input[idx + 6] * row[idx + 6] + input[idx + 14] * row[idx + 14];
                sum7 += input[idx + 7] * row[idx + 7] + input[idx + 15] * row[idx + 15];
            }
            *out = (sum0 + sum1 + sum2 + sum3) + (sum4 + sum5 + sum6 + sum7);
        }
    }

    /// Process IMDCT with windowing and overlap-add buffer update using caller-supplied scratch buffer.
    #[inline(always)]
    pub fn process_overlap_add_scratch(
        &self,
        spectral: &[f32],
        window: &[f32],
        overlap_history: &mut [f32],
        output_pcm: &mut [f32],
        scratch_2n: &mut [f32],
    ) {
        let n = self.n;
        self.imdct(spectral, scratch_2n);

        // Apply window and overlap-add
        for i in 0..n {
            let win_sample = scratch_2n[i] * window[i];
            output_pcm[i] = win_sample + overlap_history[i];
            overlap_history[i] = scratch_2n[n + i] * window[n + i];
        }
    }

    /// Process IMDCT with windowing and overlap-add buffer update.
    pub fn process_overlap_add(
        &self,
        spectral: &[f32],
        window: &[f32],
        overlap_history: &mut [f32],
        output_pcm: &mut [f32],
    ) {
        let n = self.n;
        let mut imdct_out = vec![0.0f32; 2 * n];
        self.process_overlap_add_scratch(
            spectral,
            window,
            overlap_history,
            output_pcm,
            &mut imdct_out,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify FFT-based forward MDCT matches brute-force O(N²) direct cosine matrix multiplication.
    #[test]
    fn test_forward_mdct_fft_matches_brute_force() {
        for &n in &[128, 256, 512, 1024] {
            let mdct = MdctContext::new(n);
            let two_n = 2 * n;

            // Generate test signal
            let time_in: Vec<f32> = (0..two_n)
                .map(|i| ((i as f32 * 0.037).sin() * 10000.0 + (i as f32 * 0.13).cos() * 5000.0))
                .collect();

            // Brute-force reference
            let mut spec_ref = vec![0.0f32; n];
            let scale = 2.0 / n as f32;
            for k in 0..n {
                let mut sum = 0.0f64;
                for i in 0..two_n {
                    let angle = std::f64::consts::PI / n as f64
                        * (i as f64 + 0.5 + n as f64 / 2.0)
                        * (k as f64 + 0.5);
                    sum += time_in[i] as f64 * angle.cos();
                }
                spec_ref[k] = (sum * scale as f64) as f32;
            }

            // FFT-based
            let mut spec_fft = vec![0.0f32; n];
            let mut scratch = vec![Complex32::new(0.0, 0.0); two_n];
            mdct.forward_mdct_fft(&time_in, &mut spec_fft, &mut scratch);

            // Compare
            let mut max_err = 0.0f32;
            for k in 0..n {
                let err = (spec_fft[k] - spec_ref[k]).abs();
                if err > max_err {
                    max_err = err;
                }
            }

            let max_abs = spec_ref.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let rel_err = max_err / max_abs.max(1e-10);

            assert!(
                rel_err < 1e-4,
                "FFT MDCT mismatch for N={}: max_err={:.6e}, max_abs={:.2}, rel_err={:.6e}",
                n, max_err, max_abs, rel_err
            );
        }
    }
}
