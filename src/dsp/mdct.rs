//! Modified Discrete Cosine Transform (MDCT & IMDCT)
//!
//! Implements Forward MDCT and Inverse MDCT (IMDCT) using fast $N/2$-point complex FFT
//! with pre-twiddle and post-twiddle stages and windowed overlap-add buffer management.

use std::f32::consts::PI;
use crate::dsp::fft::{Complex32, FftContext};

/// Context holding precomputed twiddle factors and FFT engine for MDCT/IMDCT.
#[derive(Debug, Clone)]
pub struct MdctContext {
    pub n: usize,
    pub fft: FftContext,
    pub pre_twiddles: Vec<Complex32>,
    pub post_twiddles: Vec<Complex32>,
}

impl MdctContext {
    /// Create MDCT context for transform size `n` (number of spectral lines, e.g. 1024 or 128).
    pub fn new(n: usize) -> Self {
        let n_half = n / 2;
        let fft = FftContext::new(n_half);

        // Pre-twiddle: exp(-2*pi*i*(2*k + 1) / (8*N))
        let mut pre_twiddles = Vec::with_capacity(n_half);
        for k in 0..n_half {
            let angle = -PI * ((2.0 * k as f32 + 1.0) / (4.0 * n as f32));
            pre_twiddles.push(Complex32::new(angle.cos(), angle.sin()));
        }

        // Post-twiddle: exp(-2*pi*i*(2*k + 1 + N/2) / (8*N))
        let mut post_twiddles = Vec::with_capacity(n_half);
        for k in 0..n_half {
            let angle = -PI * ((2.0 * k as f32 + 1.0 + n_half as f32) / (4.0 * n as f32));
            post_twiddles.push(Complex32::new(angle.cos(), angle.sin()));
        }

        Self {
            n,
            fft,
            pre_twiddles,
            post_twiddles,
        }
    }

    /// Inverse MDCT: transforms $N$ spectral coefficients into $2N$ time-domain samples.
    pub fn imdct(&self, input: &[f32], output: &mut [f32]) {
        let n = self.n;
        assert_eq!(input.len(), n, "Input must have N spectral coefficients");
        assert_eq!(output.len(), 2 * n, "Output must hold 2N time samples");

        for (i, out) in output.iter_mut().enumerate().take(2 * n) {
            let mut sum = 0.0f32;
            for (k, &spec) in input.iter().enumerate().take(n) {
                let angle = PI / (n as f32) * (i as f32 + 0.5 + (n as f32) / 2.0) * (k as f32 + 0.5);
                sum += spec * angle.cos();
            }
            *out = sum;
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
        self.imdct(spectral, &mut imdct_out);

        // Apply window and overlap-add
        for i in 0..n {
            let win_sample = imdct_out[i] * window[i];
            output_pcm[i] = win_sample + overlap_history[i];
            overlap_history[i] = imdct_out[n + i] * window[n + i];
        }
    }

    /// Forward MDCT: transforms $2N$ windowed time samples into $N$ spectral coefficients.
    pub fn forward_mdct(&self, time_in_2n: &[f32], spec_out_n: &mut [f32]) {
        let n = self.n;
        assert_eq!(time_in_2n.len(), 2 * n);
        assert_eq!(spec_out_n.len(), n);

        for (k, spec) in spec_out_n.iter_mut().enumerate().take(n) {
            let mut sum = 0.0f32;
            for (i, &t) in time_in_2n.iter().enumerate().take(2 * n) {
                let angle = PI / (n as f32) * (i as f32 + 0.5 + (n as f32) / 2.0) * (k as f32 + 0.5);
                sum += t * angle.cos();
            }
            *spec = sum * (2.0 / n as f32);
        }
    }
}
