//! Parametric Stereo (PS) Encoder Subsystem
//!
//! Extracts Inter-channel Intensity Differences (IID) and Inter-channel Coherence (ICC)
//! from a stereo pair and produces a mono downmix (ISO/IEC 14496-3 Part 3 Subpart 8).

use crate::decoder::ps::PsFrameData;
use crate::error::Result;

/// PS Encoder instance.
pub struct PsEncoder;

impl PsEncoder {
    pub fn new() -> Self {
        Self
    }

    /// Extract PS parameters and generate a mono downmix frame.
    pub fn encode_stereo(
        &self,
        left: &[f32],
        right: &[f32],
        mono_downmix: &mut [f32],
    ) -> Result<PsFrameData> {
        assert_eq!(left.len(), right.len());
        assert_eq!(left.len(), mono_downmix.len());

        let mut left_energy = 0.0f32;
        let mut right_energy = 0.0f32;
        let mut cross_corr = 0.0f32;

        for i in 0..left.len() {
            let l = left[i];
            let r = right[i];
            mono_downmix[i] = 0.5 * (l + r);
            left_energy += l * l;
            right_energy += r * r;
            cross_corr += l * r;
        }

        // Calculate IID index
        let iid_ratio = (left_energy + 1e-6) / (right_energy + 1e-6);
        let iid_db = 10.0 * iid_ratio.log10();
        let iid_idx = (iid_db * 0.5).round().clamp(-7.0, 7.0) as i8;

        // Calculate ICC index
        let denom = (left_energy * right_energy + 1e-6).sqrt();
        let coherence = (cross_corr / denom).clamp(0.0, 1.0);
        let icc_idx = ((1.0 - coherence) * 7.0).round().clamp(0.0, 7.0) as u8;

        Ok(PsFrameData {
            enable_iid: true,
            enable_icc: true,
            enable_ipd: false,
            iid_indices: vec![iid_idx],
            icc_indices: vec![icc_idx],
        })
    }
}

impl Default for PsEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ps_encoder_downmix_and_extraction() {
        let ps_enc = PsEncoder::new();
        let left = vec![1.0f32; 512];
        let right = vec![0.5f32; 512];
        let mut downmix = vec![0.0f32; 512];

        let data = ps_enc.encode_stereo(&left, &right, &mut downmix).unwrap();
        assert_eq!(downmix[0], 0.75);
        assert!(!data.iid_indices.is_empty());
    }
}
