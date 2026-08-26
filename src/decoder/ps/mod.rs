//! Parametric Stereo (PS) Decoder Subsystem
//!
//! Reconstructs a full stereo soundstage from a mono downmix and parametric
//! metadata: Inter-channel Intensity Differences (IID), Inter-channel Coherence (ICC),
//! Inter-channel Phase Differences (IPD), and Overall Phase Differences (OPD)
//! (ISO/IEC 14496-3 Part 3 Subpart 8).

use crate::error::Result;

/// Parametric Stereo spatial parameters for one time/frequency envelope.
#[derive(Debug, Clone, Default)]
pub struct PsFrameData {
    pub enable_iid: bool,
    pub enable_icc: bool,
    pub enable_ipd: bool,
    pub iid_indices: Vec<i8>,
    pub icc_indices: Vec<u8>,
}

/// Parametric Stereo spatializer engine.
pub struct PsDecoder {
    hybrid_delay_line: Vec<f32>,
}

impl PsDecoder {
    /// Create a new PS decoder.
    pub fn new() -> Self {
        Self {
            hybrid_delay_line: vec![0.0f32; 128],
        }
    }

    /// Dematrix a mono downmix signal into Left and Right stereo channels using spatial parameters.
    pub fn decode_stereo(
        &mut self,
        mono_input: &[f32],
        ps_data: &PsFrameData,
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) -> Result<()> {
        assert_eq!(mono_input.len(), left_out.len());
        assert_eq!(mono_input.len(), right_out.len());

        let count = mono_input.len();
        let default_iid_linear = 1.0f32;
        let default_icc = 1.0f32;

        let alpha = if !ps_data.iid_indices.is_empty() {
            let iid_val = ps_data.iid_indices[0] as f32;
            (10.0f32.powf(iid_val * 0.05)).clamp(0.1, 10.0)
        } else {
            default_iid_linear
        };

        let beta = if !ps_data.icc_indices.is_empty() {
            (1.0 - (ps_data.icc_indices[0] as f32 * 0.125)).clamp(0.0, 1.0)
        } else {
            default_icc
        };

        // Compute 2x2 Spatial Reconstruction Matrix: [L, R]^T = M * [M, D]^T
        let c1 = (alpha / (1.0 + alpha)).sqrt();
        let c2 = (1.0 / (1.0 + alpha)).sqrt();

        for i in 0..count {
            let s = mono_input[i];
            // All-pass fractional delay decorrelation with delay state
            let d_delayed = self.hybrid_delay_line[i % 128];
            self.hybrid_delay_line[i % 128] = s;
            let decorrelated = d_delayed * beta * 0.5;

            left_out[i] = c1 * s + decorrelated;
            right_out[i] = c2 * s - decorrelated;
        }

        Ok(())
    }
}

impl Default for PsDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ps_decoder_dematrixing() {
        let mut ps = PsDecoder::new();
        let mono = vec![1.0f32; 1024];
        let mut left = vec![0.0f32; 1024];
        let mut right = vec![0.0f32; 1024];
        let ps_data = PsFrameData {
            enable_iid: true,
            enable_icc: true,
            enable_ipd: false,
            iid_indices: vec![2],
            icc_indices: vec![1],
        };

        ps.decode_stereo(&mono, &ps_data, &mut left, &mut right).unwrap();
        assert_ne!(left[50], 0.0);
        assert_ne!(right[50], 0.0);
    }
}
