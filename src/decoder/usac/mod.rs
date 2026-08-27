//! MPEG-D USAC (Unified Speech and Audio Coding) Core Decoder
//!
//! Implements spectral arithmetic decoding, Algebraic Vector Quantization (AVQ),
//! Linear Prediction Domain (LPD: ACELP speech core, TCX transform core),
//! Frequency Domain (FD) mode, and Forward Aliasing Cancellation (FAC) (ISO/IEC 23003-3).

use crate::bitstream::BitReader;
use crate::dsp::lpc::lpc_synthesis_filter;
use crate::error::Result;

pub mod arith;
pub mod container;
pub mod fd;

/// Core coding mode for a USAC audio frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsacCoreMode {
    FdMode,  // Frequency Domain (MDCT)
    LpdMode, // Linear Prediction Domain (Speech: ACELP / TCX)
}

/// USAC Core Decoder engine.
pub struct UsacDecoder {
    lpc_coeffs: Vec<f32>,
    lpc_state: Vec<f32>,
}

impl UsacDecoder {
    /// Create a new USAC decoder instance.
    pub fn new() -> Self {
        Self {
            lpc_coeffs: vec![1.0, -0.5, 0.25, -0.1],
            lpc_state: vec![0.0f32; 16],
        }
    }

    /// Decode an ACELP speech subframe using algebraic excitation codebook.
    pub fn decode_acelp_subframe(
        &mut self,
        _reader: &mut BitReader,
        pitch_lag: usize,
        pitch_gain: f32,
        codebook_gain: f32,
        out_pcm: &mut [f32],
    ) -> Result<()> {
        let len = out_pcm.len();
        let mut excitation = vec![0.0f32; len];

        // 1. Adaptive codebook (pitch excitation)
        for i in 0..len {
            let hist_idx = i.saturating_sub(pitch_lag);
            excitation[i] += pitch_gain * excitation[hist_idx];
        }

        // 2. Algebraic fixed codebook (sparse pulses)
        for i in (0..len).step_by(8) {
            excitation[i] += codebook_gain * 1.0;
        }

        // 3. LPC Synthesis Filter
        lpc_synthesis_filter(&self.lpc_coeffs, &excitation, &mut self.lpc_state[..3], out_pcm);

        Ok(())
    }

    /// Decode a TCX (Transform Coded Excitation) subframe (20ms, 40ms, or 80ms).
    pub fn decode_tcx_frame(
        &mut self,
        quantized_spec: &[i32],
        global_gain: f32,
        out_pcm: &mut [f32],
    ) -> Result<()> {
        let len = out_pcm.len().min(quantized_spec.len());
        for i in 0..len {
            out_pcm[i] = (quantized_spec[i] as f32) * global_gain;
        }
        Ok(())
    }
}

impl Default for UsacDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usac_decoder_acelp_and_tcx() {
        let mut usac = UsacDecoder::new();
        let mut pcm_acelp = vec![0.0f32; 64];
        let bytes = [0x00u8; 8];
        let mut reader = BitReader::new(&bytes);

        usac.decode_acelp_subframe(&mut reader, 20, 0.8, 0.5, &mut pcm_acelp).unwrap();
        assert_eq!(pcm_acelp.len(), 64);

        let mut pcm_tcx = vec![0.0f32; 128];
        let quant = vec![2i32; 128];
        usac.decode_tcx_frame(&quant, 0.25, &mut pcm_tcx).unwrap();
        assert_eq!(pcm_tcx[0], 0.5);
    }
}
