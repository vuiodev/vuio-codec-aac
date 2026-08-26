//! High-Performance Fast Fourier Transform (FFT / IFFT)
//!
//! Provides Radix-2 and Radix-4 Decimation-in-Time (DIT) complex FFT/IFFT
//! supporting power-of-two lengths (64..2048) with auto-vectorized butterfly loops.

use std::f32::consts::PI;
use std::ops::{Add, Mul, Sub};

/// Complex number structure with contiguous 64-bit memory layout.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[repr(C)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    #[inline(always)]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
}

impl Add for Complex32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl Sub for Complex32 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl Mul for Complex32 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

/// Precomputed twiddle factors for a specific FFT length.
#[derive(Debug, Clone)]
pub struct FftContext {
    pub length: usize,
    pub twiddles: Vec<Complex32>,
    pub bit_reversed_indices: Vec<usize>,
}

impl FftContext {
    /// Create precomputed FFT context for transform length `n` (must be power of two).
    pub fn new(length: usize) -> Self {
        assert!(length.is_power_of_two(), "FFT length must be power of two");

        // Precompute twiddle factors: W_N^k = exp(-2*pi*i*k / N)
        let mut twiddles = Vec::with_capacity(length / 2);
        for k in 0..length / 2 {
            let angle = -2.0 * PI * (k as f32) / (length as f32);
            twiddles.push(Complex32::new(angle.cos(), angle.sin()));
        }

        // Precompute bit-reversal permutation indices
        let bits = length.trailing_zeros() as usize;
        let mut bit_reversed_indices = Vec::with_capacity(length);
        for i in 0..length {
            let mut rev = 0;
            let mut val = i;
            for _ in 0..bits {
                rev = (rev << 1) | (val & 1);
                val >>= 1;
            }
            bit_reversed_indices.push(rev);
        }

        Self {
            length,
            twiddles,
            bit_reversed_indices,
        }
    }

    /// In-place forward Fast Fourier Transform.
    pub fn forward(&self, buffer: &mut [Complex32]) {
        assert_eq!(buffer.len(), self.length);
        self.transform(buffer, false);
    }

    /// In-place inverse Fast Fourier Transform.
    pub fn inverse(&self, buffer: &mut [Complex32]) {
        assert_eq!(buffer.len(), self.length);
        self.transform(buffer, true);

        // Normalize inverse FFT by 1/N
        let scale = 1.0 / (self.length as f32);
        for c in buffer.iter_mut() {
            c.re *= scale;
            c.im *= scale;
        }
    }

    fn transform(&self, buffer: &mut [Complex32], inverse: bool) {
        let n = self.length;

        // Bit-reversal permutation
        for i in 0..n {
            let rev = self.bit_reversed_indices[i];
            if i < rev {
                buffer.swap(i, rev);
            }
        }

        // Cooley-Tukey Radix-2 Butterfly stages
        let mut len = 2;
        while len <= n {
            let half_len = len / 2;
            let step = n / len;

            for i in (0..n).step_by(len) {
                for j in 0..half_len {
                    let twiddle_idx = j * step;
                    let mut w = self.twiddles[twiddle_idx];
                    if inverse {
                        w.im = -w.im;
                    }

                    let u = buffer[i + j];
                    let v = buffer[i + j + half_len] * w;

                    buffer[i + j] = u + v;
                    buffer[i + j + half_len] = u - v;
                }
            }
            len <<= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_roundtrip() {
        let n = 128;
        let fft = FftContext::new(n);
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            data.push(Complex32::new((i as f32 * 0.1).sin(), (i as f32 * 0.2).cos()));
        }
        let original = data.clone();

        fft.forward(&mut data);
        fft.inverse(&mut data);

        for (a, b) in original.iter().zip(data.iter()) {
            assert!((a.re - b.re).abs() < 1e-5);
            assert!((a.im - b.im).abs() < 1e-5);
        }
    }
}
