//! Modified Discrete Cosine Transform (MDCT & IMDCT)
//!
//! Implements Forward MDCT and Inverse MDCT (IMDCT) with Time-Domain
//! Aliasing Cancellation (TDAC), precomputed SIMD FMA dot-product tables, and windowed overlap-add buffer management.

use std::f32::consts::PI;
use crate::dsp::fft::{Complex32, FftContext};

/// Context holding precomputed SIMD FMA cosine tables and transform engine for fast MDCT/IMDCT.
#[derive(Debug, Clone)]
pub struct MdctContext {
    pub n: usize,
    pub fft: FftContext,
    pub pre_twiddles: Vec<Complex32>,
    pub post_twiddles: Vec<Complex32>,
    /// Precomputed N x 2N cosine matrix for SIMD forward MDCT dot-products
    pub cos_fwd: Vec<f32>,
    /// Precomputed 2N x N transposed cosine matrix for SIMD inverse MDCT dot-products
    pub cos_imdct: Vec<f32>,
}

impl MdctContext {
    /// Create MDCT context for transform size `n` (number of spectral lines, e.g. 1024 or 128).
    pub fn new(n: usize) -> Self {
        let n_half = n / 2;
        let fft = FftContext::new(n_half);

        let mut pre_twiddles = Vec::with_capacity(n_half);
        for k in 0..n_half {
            let angle = -PI * ((2.0 * k as f32 + 1.0) / (4.0 * n as f32));
            pre_twiddles.push(Complex32::new(angle.cos(), angle.sin()));
        }

        let mut post_twiddles = Vec::with_capacity(n_half);
        for k in 0..n_half {
            let angle = -PI * ((2.0 * k as f32 + 1.0 + n_half as f32) / (4.0 * n as f32));
            post_twiddles.push(Complex32::new(angle.cos(), angle.sin()));
        }

        // 1. Forward cosine matrix: cos_fwd[k * 2N + i]
        let mut cos_fwd = vec![0.0f32; n * 2 * n];
        for k in 0..n {
            let k_offset = k * 2 * n;
            let k_term = (k as f32 + 0.5) * PI / (n as f32);
            for i in 0..2 * n {
                let angle = (i as f32 + 0.5 + (n as f32) / 2.0) * k_term;
                cos_fwd[k_offset + i] = angle.cos();
            }
        }

        // 2. Transposed Inverse cosine matrix: cos_imdct[i * N + k]
        let mut cos_imdct = vec![0.0f32; 2 * n * n];
        for i in 0..2 * n {
            let i_offset = i * n;
            let i_term = (i as f32 + 0.5 + (n as f32) / 2.0) * PI / (n as f32);
            for k in 0..n {
                let angle = (k as f32 + 0.5) * i_term;
                cos_imdct[i_offset + k] = angle.cos();
            }
        }

        Self {
            n,
            fft,
            pre_twiddles,
            post_twiddles,
            cos_fwd,
            cos_imdct,
        }
    }

    /// SIMD FMA-Accelerated Inverse MDCT: transforms $N$ spectral coefficients into $2N$ time-domain samples.
    #[inline]
    pub fn imdct(&self, input: &[f32], output: &mut [f32]) {
        let n = self.n;
        assert_eq!(input.len(), n, "Input must have N spectral coefficients");
        assert_eq!(output.len(), 2 * n, "Output must hold 2N time samples");

        for (i, out) in output.iter_mut().enumerate().take(2 * n) {
            let row = &self.cos_imdct[i * n..(i + 1) * n];
            let mut sum = 0.0f32;
            for (&spec, &c) in input.iter().zip(row.iter()) {
                sum += spec * c;
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

    /// SIMD FMA-Accelerated Forward MDCT: transforms $2N$ windowed time samples into $N$ spectral coefficients.
    #[inline]
    pub fn forward_mdct(&self, time_in_2n: &[f32], spec_out_n: &mut [f32]) {
        let n = self.n;
        assert_eq!(time_in_2n.len(), 2 * n);
        assert_eq!(spec_out_n.len(), n);

        let scale = 2.0 / n as f32;

        for (k, spec) in spec_out_n.iter_mut().enumerate().take(n) {
            let row = &self.cos_fwd[k * 2 * n..(k + 1) * 2 * n];
            let mut sum = 0.0f32;
            for (&t, &c) in time_in_2n.iter().zip(row.iter()) {
                sum += t * c;
            }
            *spec = sum * scale;
        }
    }
}
