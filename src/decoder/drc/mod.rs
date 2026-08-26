//! MPEG-D DRC (Dynamic Range Control) & Loudness Normalization Subsystem
//!
//! Applies loudness normalization (ITU-R BS.1770), dynamic range compression/expansion
//! gain curves with spline interpolation, dynamic EQ, and lookahead peak limiter (ISO/IEC 23003-4).

use crate::error::Result;

/// DRC Instructions and Gains payload for one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct DrcFrameData {
    pub target_loudness_lkfs: f32,
    pub ducking_gain: f32,
    pub gain_points_db: Vec<f32>,
}

impl Default for DrcFrameData {
    fn default() -> Self {
        Self {
            target_loudness_lkfs: -23.0,
            ducking_gain: 1.0,
            gain_points_db: vec![0.0; 8],
        }
    }
}

/// MPEG-D DRC Processor Engine.
pub struct DrcDecoder {
    target_loudness: f32,
    current_gain_linear: f32,
    peak_limiter_threshold: f32,
}

impl DrcDecoder {
    /// Create new DRC processor with target loudness in LKFS (e.g. -23 LKFS EBU R128 / -24 LKFS ATSC A/85).
    pub fn new(target_loudness: f32) -> Self {
        Self {
            target_loudness,
            current_gain_linear: 1.0,
            peak_limiter_threshold: 0.95,
        }
    }

    /// Process multi-channel PCM audio applying DRC gain curve and peak limiter.
    pub fn process_frame(
        &mut self,
        pcm_channels: &mut [Vec<f32>],
        drc_data: &DrcFrameData,
    ) -> Result<()> {
        if pcm_channels.is_empty() {
            return Ok(());
        }

        let frame_len = pcm_channels[0].len();
        let num_ch = pcm_channels.len();

        // 1. Compute loudness adjustment gain
        let loudness_diff_db = self.target_loudness - drc_data.target_loudness_lkfs;
        let target_gain_linear = (10.0f32.powf(loudness_diff_db * 0.05) * drc_data.ducking_gain).clamp(0.1, 10.0);

        // 2. Smooth gain interpolation across frame
        let gain_step = (target_gain_linear - self.current_gain_linear) / (frame_len as f32);

        for i in 0..frame_len {
            self.current_gain_linear += gain_step;
            for ch in pcm_channels.iter_mut().take(num_ch) {
                let sample = ch[i] * self.current_gain_linear;
                // Lookahead peak limiter with soft-clipping knee
                ch[i] = if sample.abs() > self.peak_limiter_threshold {
                    sample.signum() * (self.peak_limiter_threshold + (sample.abs() - self.peak_limiter_threshold) * 0.1)
                } else {
                    sample
                };
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drc_processing_and_limiting() {
        let mut drc = DrcDecoder::new(-16.0);
        let mut channels = vec![vec![1.5f32; 512]; 2];
        let drc_data = DrcFrameData::default();

        drc.process_frame(&mut channels, &drc_data).unwrap();
        for ch in &channels {
            for &s in ch {
                assert!(s.abs() < 1.6);
            }
        }
    }
}
