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
        let n_half = n / 2;
        assert_eq!(input.len(), n, "Input must have N spectral coefficients");
        assert_eq!(output.len(), 2 * n, "Output must hold 2N time samples");

        // 1. Pre-twiddle modulation into N/2 complex buffer
        let mut fft_buf = vec![Complex32::default(); n_half];
        for k in 0..n_half {
            let re = -input[2 * (n_half - 1 - k) + 1];
            let im = input[2 * k];
            let twiddle = self.pre_twiddles[k];
            fft_buf[k] = Complex32::new(
                re * twiddle.re - im * twiddle.im,
                re * twiddle.im + im * twiddle.re,
            );
        }

        // 2. N/2 point IFFT
        self.fft.forward(&mut fft_buf);

        // 3. Post-twiddle and time-domain unfolding into 2N samples
        let mut unfolded = vec![0.0f32; n];
        for k in 0..n_half {
            let twiddle = self.post_twiddles[k];
            let c = fft_buf[k];
            let re = c.re * twiddle.re - c.im * twiddle.im;
            let im = c.re * twiddle.im + c.im * twiddle.re;
            unfolded[2 * k] = im;
            unfolded[2 * k + 1] = -re;
        }

        // 4. Construct the 2N output waveform with symmetry unfolding
        for i in 0..n_half {
            output[i] = -unfolded[n_half + i];
            output[n_half + i] = -unfolded[n - 1 - i];
            output[n + i] = unfolded[i];
            output[n + n_half + i] = unfolded[n_half - 1 - i];
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
}
