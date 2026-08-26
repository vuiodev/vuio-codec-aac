//! Perceptual Noise Substitution (PNS) Subsystem
//!
//! Generates pseudo-random white noise vectors scaled by band energy
//! to replace noise-like spectral bands efficiently.

/// Pseudo-random noise generator using 32-bit Linear Congruential Generator (LCG).
#[derive(Debug, Clone)]
pub struct PnsGenerator {
    seed: u32,
}

impl Default for PnsGenerator {
    fn default() -> Self {
        Self::new(0x03030303)
    }
}

impl PnsGenerator {
    pub const fn new(seed: u32) -> Self {
        Self { seed }
    }

    /// Generate next pseudo-random float in range `[-1.0, 1.0]`.
    pub fn next_sample(&mut self) -> f32 {
        // Standard MPEG AAC LCG: seed = (seed * 1664525 + 1013904223)
        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let signed_val = (self.seed >> 16) as i16;
        (signed_val as f32) / 32768.0
    }

    /// Fill target band with normalized noise scaled by scalefactor energy.
    pub fn fill_noise_band(&mut self, scalefactor: i16, output: &mut [f32]) {
        let n = output.len();
        if n == 0 {
            return;
        }

        let sf_shift = (scalefactor as f32 - 100.0) * 0.25;
        let target_energy = 2.0f32.powf(sf_shift);

        let mut energy_sum = 0.0f32;
        for s in output.iter_mut() {
            let sample = self.next_sample();
            *s = sample;
            energy_sum += sample * sample;
        }

        if energy_sum > 1e-12 {
            let scale = (target_energy / energy_sum.sqrt()) / (n as f32).sqrt();
            for s in output.iter_mut() {
                *s *= scale;
            }
        }
    }
}
