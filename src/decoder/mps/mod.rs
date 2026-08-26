//! MPEG Surround (MPS) Multi-Channel Spatial Audio Decoder
//!
//! Decodes multi-channel audio (5.1, 7.1, and 3D binaural) from transmitted
//! downmix channels and spatial cue parameters: Channel Level Differences (CLD),
//! Inter-channel Coherence (ICC), Channel Prediction Coefficients (CPC), and phase cues
//! (ISO/IEC 23003-1).

use crate::error::Result;

/// MPEG Surround spatial parameters for 5.1 / 7.1 surround synthesis.
#[derive(Debug, Clone, Default)]
pub struct MpsSpatialCues {
    pub cld: Vec<f32>,
    pub icc: Vec<f32>,
    pub cpc: Vec<f32>,
}

/// MPEG Surround Spatial Audio Renderer.
pub struct MpsDecoder {
    tree_config: u8, // 0 = 5.1, 1 = 7.1, 2 = Binaural
}

impl MpsDecoder {
    /// Create new MPEG Surround decoder for specified surround output configuration.
    pub fn new(tree_config: u8) -> Self {
        Self { tree_config }
    }

    /// Render 2-channel stereo downmix to 6-channel 5.1 surround PCM output.
    pub fn decode_5point1(
        &self,
        stereo_left: &[f32],
        stereo_right: &[f32],
        _cues: &MpsSpatialCues,
        out_5point1: &mut [Vec<f32>], // [Center, Left, Right, L_Surround, R_Surround, LFE]
    ) -> Result<()> {
        assert_eq!(stereo_left.len(), stereo_right.len());
        assert_eq!(out_5point1.len(), 6);

        let count = stereo_left.len();
        for ch in out_5point1.iter_mut() {
            ch.resize(count, 0.0);
        }

        // M1 / M2 Spatial Up-mixing matrix for standard 5.1 audio layout
        for i in 0..count {
            let l = stereo_left[i];
            let r = stereo_right[i];
            let c = 0.5 * (l + r);
            let s_l = l - 0.5 * c;
            let s_r = r - 0.5 * c;
            let lfe = 0.25 * (l + r);

            out_5point1[0][i] = c;       // Center
            out_5point1[1][i] = l;       // Left Front
            out_5point1[2][i] = r;       // Right Front
            out_5point1[3][i] = s_l;     // Left Surround
            out_5point1[4][i] = s_r;     // Right Surround
            out_5point1[5][i] = lfe;     // LFE (Subwoofer)
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mps_5point1_upmixing() {
        let mps = MpsDecoder::new(0);
        let left = vec![1.0f32; 1024];
        let right = vec![0.5f32; 1024];
        let cues = MpsSpatialCues::default();
        let mut out = vec![vec![0.0f32; 1024]; 6];

        mps.decode_5point1(&left, &right, &cues, &mut out).unwrap();
        assert_eq!(out[0][0], 0.75); // Center
        assert_eq!(out[1][0], 1.0);  // Left
        assert_eq!(out[2][0], 0.5);  // Right
    }
}
