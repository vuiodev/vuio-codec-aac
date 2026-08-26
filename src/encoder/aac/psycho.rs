//! Psychoacoustic Model and Masking Threshold Analysis
//!
//! Calculates tonality, energy per scalefactor band, and spreading functions
//! to determine Perceptual Entropy (PE) and masking thresholds for rate control with zero allocations.

/// Perceptual analysis result for a single channel frame.
#[derive(Debug, Clone)]
pub struct PsychoResult {
    pub energy_per_band: [f32; 64],
    pub masking_thresholds: [f32; 64],
    pub perceptual_entropy: f32,
    pub num_bands: usize,
}

/// Psychoacoustic analyzer instance with contiguous flat memory layout.
#[derive(Debug, Clone)]
pub struct PsychoacousticModel {
    num_bands: usize,
    flat_spreading_matrix: Vec<f32>,
}

impl PsychoacousticModel {
    /// Create new psychoacoustic analyzer for specified number of scalefactor bands.
    pub fn new(num_bands: usize) -> Self {
        let mut flat_spreading_matrix = vec![0.0f32; num_bands * num_bands];
        for i in 0..num_bands {
            let row_offset = i * num_bands;
            for j in 0..num_bands {
                let bark_diff = (i as f32 - j as f32).abs();
                flat_spreading_matrix[row_offset + j] = (-bark_diff * 0.4).exp();
            }
        }

        Self {
            num_bands,
            flat_spreading_matrix,
        }
    }

    /// Analyze spectral energy and compute masking thresholds with zero heap allocations.
    #[inline(always)]
    pub fn analyze(&self, spectral: &[f32], sfb_offsets: &[usize]) -> PsychoResult {
        let num_bands = (sfb_offsets.len().saturating_sub(1)).min(self.num_bands).min(64);
        let mut energy_per_band = [0.0f32; 64];
        let mut spread_energy = [0.0f32; 64];
        let mut masking_thresholds = [0.0f32; 64];

        // 1. Calculate energy per SFB
        for b in 0..num_bands {
            let start = sfb_offsets[b];
            let end = sfb_offsets[b + 1].min(spectral.len());
            let mut energy = 0.0f32;
            for &c in &spectral[start..end] {
                energy += c * c;
            }
            energy_per_band[b] = energy;
        }

        // 2. SIMD contiguous convolution with flat spreading function
        for i in 0..num_bands {
            let row = &self.flat_spreading_matrix[i * self.num_bands..i * self.num_bands + num_bands];
            let mut sum = 0.0f32;
            for (&energy, &weight) in energy_per_band[..num_bands].iter().zip(row.iter()) {
                sum += energy * weight;
            }
            spread_energy[i] = sum;
        }

        // 3. Masking thresholds & Perceptual Entropy (PE)
        let mut pe = 0.0f32;
        for b in 0..num_bands {
            let threshold = (spread_energy[b] * 0.29).max(1e-5);
            masking_thresholds[b] = threshold;

            let nb_lines = (sfb_offsets[b + 1] - sfb_offsets[b]) as f32;
            let ratio = (energy_per_band[b] / threshold).max(1.0);
            pe += nb_lines * ratio.log2();
        }

        PsychoResult {
            energy_per_band,
            masking_thresholds,
            perceptual_entropy: pe,
            num_bands,
        }
    }
}
