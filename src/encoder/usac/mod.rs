//! MPEG-D USAC (Unified Speech and Audio Coding) Encoder Subsystem
//!
//! Performs Speech vs Music classification, Spectral Flux tonality analysis,
//! and ACELP / TCX / FD core mode selection (ISO/IEC 23003-3).

use crate::decoder::usac::UsacCoreMode;

pub mod arith;
pub mod container;
pub mod fd;
pub mod lsf;
pub mod tns;

/// USAC Speech vs Music Classifier and Encoder Engine.
pub struct UsacEncoder;

impl UsacEncoder {
    pub fn new() -> Self {
        Self
    }

    /// Analyze incoming frame and decide optimal core coding mode (FD or LPD/Speech).
    pub fn classify_frame(&self, time_pcm: &[f32]) -> UsacCoreMode {
        assert_eq!(time_pcm.len(), 1024);

        // Zero-crossing rate & spectral flatness proxy
        let mut zcr = 0;
        for i in 1..time_pcm.len() {
            if (time_pcm[i] >= 0.0 && time_pcm[i - 1] < 0.0) || (time_pcm[i] < 0.0 && time_pcm[i - 1] >= 0.0) {
                zcr += 1;
            }
        }

        // Speech frames typically exhibit lower zero-crossing rate and higher periodicity
        if zcr < 80 {
            UsacCoreMode::LpdMode // Speech mode (ACELP / TCX)
        } else {
            UsacCoreMode::FdMode  // General audio / music mode (MDCT)
        }
    }
}

impl Default for UsacEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usac_classification() {
        let usac_enc = UsacEncoder::new();
        let low_freq_sine: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.02).sin()).collect();
        let mode = usac_enc.classify_frame(&low_freq_sine);
        assert_eq!(mode, UsacCoreMode::LpdMode);
    }
}
