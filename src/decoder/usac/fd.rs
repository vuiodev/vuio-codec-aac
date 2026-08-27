//! Minimal USAC frequency-domain (FD) single-channel-element decoder.
//!
//! The counterpart to [`crate::encoder::usac::fd`]: reads one frame's
//! `global_gain`, `scale_factor_data()` and arithmetic-coded spectral data,
//! dequantizes it, and reconstructs PCM through the same overlap-add
//! filterbank the AAC-LC decoder uses ([`Filterbank`] takes a window
//! sequence and shape purely to decide how to window the transform; feeding
//! it `OnlyLongSequence`/`Sine` here is exact, not a reused approximation —
//! this path never has anything else to feed it). See that module's docs
//! for what is and is not covered by this minimal shape.

use crate::bitstream::BitReader;
use crate::decoder::aac::huffman::decode_scalefactor_delta;
use crate::decoder::aac::ics::SF_OFFSET;
use crate::decoder::usac::arith::{decode_pairs, dequantize};
use crate::dsp::filterbank::Filterbank;
use crate::error::Result;
use crate::tables::scalefactor::{MAX_SFB_LONG, compute_sfb_offsets};
use crate::tables::sfb::SFB_48_1024;
use crate::tables::usac_arith::{Contexts, TABLE_EXP, TABLE_FRAC};
use crate::types::{WindowSequence, WindowShape};

/// Samples per frame this minimal path codes; see
/// [`crate::encoder::usac::fd::FRAME_LEN`].
pub const FRAME_LEN: usize = 1024;

/// [`dequantize`]'s fixed-point output runs six bits hotter than the plain
/// linear scale [`quantize_band`](crate::encoder::aac::quant::quantize_band)
/// works in: its magnitude table is Q13 and `fac_fix` (below) is Q15, and
/// their product is taken back down by only 22 of those 28 bits (see
/// `ixheaacd_esc_iquant`), leaving a constant factor of `2^(28-22) = 64`
/// baked into every coefficient. Dividing it back out here is what makes
/// this decoder's reconstruction land in the same units the encoder
/// quantized in, rather than a fixed-point convention that only matters
/// inside the reference decoder's own downstream stages.
const ARITH_FIXED_POINT_SHIFT: f32 = 64.0;

/// One FD single-channel element's worth of decoder state.
pub struct UsacFdDecoder {
    filterbank: Filterbank,
    overlap: Vec<f32>,
    /// Bands `scale_factor_data()` carries; see the encoder's field of the
    /// same name.
    num_sfb: usize,
    /// Arithmetic-coder context history, carried across frames.
    contexts: Contexts,
    /// Threaded through to [`dequantize`] even though this path never sets
    /// `with_noise` (noise filling is separate, larger work), since the seed
    /// still has to live somewhere for when it does.
    noise_seed: u32,
}

impl UsacFdDecoder {
    pub fn new() -> Self {
        let mut offsets = [0usize; MAX_SFB_LONG + 1];
        let count = compute_sfb_offsets(SFB_48_1024, &mut offsets);
        Self {
            filterbank: Filterbank::new(FRAME_LEN),
            overlap: vec![0.0; FRAME_LEN],
            num_sfb: count - 1,
            contexts: Contexts::new(),
            noise_seed: 0,
        }
    }

    /// Decode one frame's raw block into `FRAME_LEN` PCM samples.
    pub fn decode_frame(&mut self, reader: &mut BitReader) -> Result<Vec<f32>> {
        let global_gain = reader.read_u8(8)? as i32;
        let mut scalefactor = global_gain;
        for _ in 1..self.num_sfb {
            scalefactor += decode_scalefactor_delta(reader)?;
        }

        let pairs = FRAME_LEN / 2;
        let mut quant = vec![0i32; FRAME_LEN];
        decode_pairs(reader, &mut self.contexts, pairs, pairs, &mut quant);

        let mut coef = vec![0i32; FRAME_LEN];
        dequantize(&quant, &mut coef, 0, false, &mut self.noise_seed, fac_fix(scalefactor));

        let mut spectral = vec![0.0f32; FRAME_LEN];
        for (s, &c) in spectral.iter_mut().zip(coef.iter()) {
            *s = c as f32 / ARITH_FIXED_POINT_SHIFT;
        }

        let mut out = vec![0.0f32; FRAME_LEN];
        self.filterbank.synthesize(
            &spectral,
            WindowSequence::OnlyLongSequence,
            WindowShape::Sine,
            WindowShape::Sine,
            &mut self.overlap,
            &mut out,
        );
        Ok(out)
    }
}

impl Default for UsacFdDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Turn a scalefactor into the fixed-point linear multiplier [`dequantize`]
/// expects, mirroring `ixheaacd_apply_scfs_and_nf`'s inline computation in
/// the reference decoder: `2^((scalefactor - 100) / 4)` in Q15, split into an
/// integer power of two ([`TABLE_EXP`]) and a quarter-step fraction
/// ([`TABLE_FRAC`]).
fn fac_fix(scalefactor: i32) -> i64 {
    let fac = scalefactor - SF_OFFSET;
    if fac < 0 {
        // The reference decoder hard-zeros a band here rather than letting a
        // negative shift underflow; a real encoder never emits a scalefactor
        // this low in the first place (see the encoder module's doc).
        return 0;
    }
    let exp = (fac >> 2).min(31) as usize;
    let frac = (fac & 3) as usize;
    (TABLE_FRAC[3 + frac] as i64 * TABLE_EXP[exp]) >> 15
}
