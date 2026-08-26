//! Psychoacoustic Model and Masking Threshold Analysis
//!
//! Calculates tonality, energy per scalefactor band, and spreading functions
//! to determine Perceptual Entropy (PE) and masking thresholds for rate control.

/// Perceptual analysis result for a single channel frame.
#[derive(Debug, Clone)]
pub struct PsychoResult {
    pub energy_per_band: Vec<f32>,
    pub masking_thresholds: Vec<f32>,
    pub perceptual_entropy: f32,
}

/// Psychoacoustic analyzer instance.
#[derive(Debug, Clone)]
pub struct PsychoacousticModel {
    num_bands: usize,
    spreading_matrix: Vec<Vec<f32>>,
}

impl PsychoacousticModel {
    /// Create new psychoacoustic analyzer for specified number of scalefactor bands.
    pub fn new(num_bands: usize) -> Self {
        let mut spreading_matrix = vec![vec![0.0f32; num_bands]; num_bands];
        for (i, row) in spreading_matrix.iter_mut().enumerate().take(num_bands) {
            for (j, cell) in row.iter_mut().enumerate().take(num_bands) {
                let bark_diff = (i as f32 - j as f32).abs();
                *cell = (-bark_diff * 0.4).exp();
            }
        }

        Self {
            num_bands,
            spreading_matrix,
        }
    }

    /// Analyze spectral energy and compute masking thresholds.
    pub fn analyze(&self, spectral: &[f32], sfb_offsets: &[usize]) -> PsychoResult {
        let num_bands = (sfb_offsets.len().saturating_sub(1)).min(self.num_bands);
        let mut energy_per_band = vec![0.0f32; num_bands];

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

        // 2. Convolution with spreading function
        let mut spread_energy = vec![0.0f32; num_bands];
        for (i, spread_val) in spread_energy.iter_mut().enumerate().take(num_bands) {
            let mut sum = 0.0f32;
            for (j, &energy) in energy_per_band.iter().enumerate().take(num_bands) {
                sum += energy * self.spreading_matrix[i][j];
            }
            *spread_val = sum;
        }

        // 3. Masking thresholds & Perceptual Entropy (PE)
        let mut masking_thresholds = vec![0.0f32; num_bands];
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
        }
    }
}
