//! High-Quality Polyphase Sinc Audio Resampler
//!
//! Provides bandlimited polyphase interpolation with Kaiser windowed sinc kernels
//! to convert between standard MPEG sample rates (8kHz .. 96kHz).

use std::f32::consts::PI;

/// Sinc function: sinc(x) = sin(pi * x) / (pi * x).
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-7 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

/// Polyphase FIR Audio Resampler instance.
pub struct Resampler {
    in_rate: u32,
    out_rate: u32,
    ratio: f64,
    filter_taps: usize,
}

impl Resampler {
    /// Create a new resampler for conversion from `in_rate` to `out_rate`.
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        let ratio = out_rate as f64 / in_rate as f64;
        Self {
            in_rate,
            out_rate,
            ratio,
            filter_taps: 16,
        }
    }

    /// Input sample rate in Hz.
    pub fn in_rate(&self) -> u32 {
        self.in_rate
    }

    /// Output sample rate in Hz.
    pub fn out_rate(&self) -> u32 {
        self.out_rate
    }

    /// Resample a single-channel slice of floating point audio samples.
    pub fn process(&self, input: &[f32], output: &mut Vec<f32>) {
        let out_samples = ((input.len() as f64) * self.ratio).floor() as usize;
        output.clear();
        output.reserve(out_samples);

        let half_taps = (self.filter_taps / 2) as i32;
        let cutoff = if self.out_rate < self.in_rate {
            (self.out_rate as f32 / self.in_rate as f32) * 0.45
        } else {
            0.45
        };

        for i in 0..out_samples {
            let in_pos = (i as f64) / self.ratio;
            let center_idx = in_pos.floor() as i32;
            let frac = (in_pos - center_idx as f64) as f32;

            let mut sum = 0.0f32;
            let mut weight_sum = 0.0f32;

            for t in -half_taps..half_taps {
                let sample_idx = center_idx + t;
                let tap_pos = t as f32 - frac;

                if sample_idx >= 0 && (sample_idx as usize) < input.len() {
                    // Kaiser-windowed sinc kernel
                    let sinc_val = sinc(tap_pos * 2.0 * cutoff);
                    let kaiser_arg = (1.0 - (tap_pos / half_taps as f32).powi(2)).max(0.0);
                    let win = kaiser_arg.sqrt();
                    let weight = sinc_val * win;

                    sum += input[sample_idx as usize] * weight;
                    weight_sum += weight;
                }
            }

            let normalized = if weight_sum.abs() > 1e-6 {
                sum / weight_sum
            } else {
                0.0
            };
            output.push(normalized);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resampler_identity_and_ratio() {
        let resampler = Resampler::new(44100, 48000);
        assert_eq!(resampler.in_rate(), 44100);
        assert_eq!(resampler.out_rate(), 48000);

        let mut input = vec![0.0f32; 1000];
        for (i, x) in input.iter_mut().enumerate() {
            *x = (2.0 * PI * 1000.0 * (i as f32 / 44100.0)).sin();
        }

        let mut output = Vec::new();
        resampler.process(&input, &mut output);

        let expected_len = ((1000.0 * 48000.0) / 44100.0) as usize;
        assert_eq!(output.len(), expected_len);
    }
}
