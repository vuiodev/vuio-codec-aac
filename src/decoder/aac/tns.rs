//! Temporal Noise Shaping (TNS) Inverse Filter Subsystem
//!
//! Applies all-pole linear prediction lattice filtering across spectral lines
//! to shape quantization noise in the time domain.

/// TNS Filter parameters for a single window.
#[derive(Debug, Clone, PartialEq)]
pub struct TnsFilter {
    pub start_band: usize,
    pub stop_band: usize,
    pub order: usize,
    pub direction: bool, // false: forward, true: backward
    pub coef_res: bool,  // false: 3-bit, true: 4-bit
    pub coefficients: Vec<f32>,
}

impl TnsFilter {
    /// Convert reflection coefficients into direct-form LP filter coefficients.
    pub fn compute_lpc_coefficients(&self) -> Vec<f32> {
        let order = self.order;
        let mut lpc = vec![0.0f32; order + 1];
        lpc[0] = 1.0;

        for (i, &gamma) in self.coefficients.iter().enumerate() {
            let k = i + 1;
            let mut next_lpc = lpc.clone();
            next_lpc[k] = gamma;
            for j in 1..k {
                next_lpc[j] = lpc[j] - gamma * lpc[k - j];
            }
            lpc = next_lpc;
        }

        lpc
    }

    /// Apply inverse all-pole TNS filter in-place to spectral slice.
    pub fn apply(&self, spectrum: &mut [f32]) {
        if self.order == 0 || spectrum.is_empty() {
            return;
        }

        let lpc = self.compute_lpc_coefficients();
        let order = self.order;
        let mut state = vec![0.0f32; order];

        if !self.direction {
            // Forward filtering
            for sample in spectrum.iter_mut() {
                let mut sum = *sample;
                for j in 1..=order {
                    sum -= lpc[j] * state[order - j];
                }
                *sample = sum;
                state.copy_within(1..order, 0);
                state[order - 1] = sum;
            }
        } else {
            // Backward filtering
            for sample in spectrum.iter_mut().rev() {
                let mut sum = *sample;
                for j in 1..=order {
                    sum -= lpc[j] * state[order - j];
                }
                *sample = sum;
                state.copy_within(1..order, 0);
                state[order - 1] = sum;
            }
        }
    }
}
