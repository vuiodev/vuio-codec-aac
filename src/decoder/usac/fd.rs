//! Minimal USAC frequency-domain (FD) decoder: single-channel and stereo.
//!
//! The counterpart to [`crate::encoder::usac::fd`]: reads one frame's
//! `global_gain`, `scale_factor_data()` and arithmetic-coded spectral data,
//! dequantizes it, and reconstructs PCM through the same overlap-add
//! filterbank the AAC-LC decoder uses ([`Filterbank`] takes a window
//! sequence and shape purely to decide how to window the transform; feeding
//! it `OnlyLongSequence`/`Sine` here is exact, not a reused approximation —
//! this path never has anything else to feed it). See that module's docs
//! for what is and is not covered by this minimal shape.
//!
//! # A decode-side bug the per-band rate loop surfaced
//!
//! The single-scalefactor-per-frame version of this decoder accumulated
//! every `scale_factor_data()` delta into one final running scalar and
//! dequantized the *whole* spectrum with it in one call — which happened to
//! be harmless when the encoder only ever transmitted zero deltas after the
//! first, but is wrong the moment scalefactors genuinely vary per band: the
//! final accumulated value has nothing to do with any individual band's
//! actual scalefactor. [`UsacFdDecoder::decode_frame`] now keeps the whole
//! per-band array from [`read_scale_factor_data`] and dequantizes each
//! band's coefficient range with its own value, matching what
//! [`crate::encoder::usac::fd`]'s real per-band rate loop now transmits.

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

/// Bands [`SFB_48_1024`] resolves to; shared setup both the mono and stereo
/// decoders need.
struct Layout {
    sfb_offsets: [usize; MAX_SFB_LONG + 1],
    num_sfb: usize,
}

impl Layout {
    fn new() -> Self {
        let mut sfb_offsets = [0usize; MAX_SFB_LONG + 1];
        let count = compute_sfb_offsets(SFB_48_1024, &mut sfb_offsets);
        Self { sfb_offsets, num_sfb: count - 1 }
    }
}

/// One channel's arithmetic-coder context history and noise-fill seed.
struct ChannelState {
    filterbank: Filterbank,
    overlap: Vec<f32>,
    contexts: Contexts,
    noise_seed: u32,
}

impl ChannelState {
    fn new() -> Self {
        Self {
            filterbank: Filterbank::new(FRAME_LEN),
            overlap: vec![0.0; FRAME_LEN],
            contexts: Contexts::new(),
            noise_seed: 0,
        }
    }
}

/// Read `global_gain` followed by one Huffman-coded delta per remaining
/// band into a full per-band scalefactor array, the shape
/// [`crate::encoder::usac::fd::write_scale_factor_data`] writes.
fn read_scale_factor_data(reader: &mut BitReader, num_sfb: usize) -> Result<Vec<i32>> {
    let mut scalefactors = vec![0i32; num_sfb];
    scalefactors[0] = reader.read_u8(8)? as i32;
    for b in 1..num_sfb {
        scalefactors[b] = scalefactors[b - 1] + decode_scalefactor_delta(reader)?;
    }
    Ok(scalefactors)
}

/// Dequantize one channel's coded magnitudes band by band, each with its
/// own scalefactor, and scale the arithmetic coder's fixed-point output
/// down into the same linear units the encoder quantized in.
fn dequantize_channel(
    quant: &[i32],
    sfb_offsets: &[usize],
    num_sfb: usize,
    scalefactors: &[i32],
    noise_seed: &mut u32,
    spectral: &mut [f32],
) {
    let mut coef = vec![0i32; quant.len()];
    for b in 0..num_sfb {
        let lo = sfb_offsets[b];
        let hi = sfb_offsets[b + 1].min(quant.len());
        if lo >= hi {
            continue;
        }
        dequantize(&quant[lo..hi], &mut coef[lo..hi], 0, false, noise_seed, fac_fix(scalefactors[b]));
    }
    for (s, &c) in spectral.iter_mut().zip(coef.iter()) {
        *s = c as f32 / ARITH_FIXED_POINT_SHIFT;
    }
}

/// One FD single-channel element's worth of decoder state.
pub struct UsacFdDecoder {
    layout: Layout,
    channel: ChannelState,
}

impl UsacFdDecoder {
    pub fn new() -> Self {
        Self { layout: Layout::new(), channel: ChannelState::new() }
    }

    /// Decode one frame's raw block into `FRAME_LEN` PCM samples.
    pub fn decode_frame(&mut self, reader: &mut BitReader) -> Result<Vec<f32>> {
        let num_sfb = self.layout.num_sfb;
        let scalefactors = read_scale_factor_data(reader, num_sfb)?;

        let pairs = FRAME_LEN / 2;
        let mut quant = vec![0i32; FRAME_LEN];
        decode_pairs(reader, &mut self.channel.contexts, pairs, pairs, &mut quant);

        let mut spectral = vec![0.0f32; FRAME_LEN];
        dequantize_channel(
            &quant,
            &self.layout.sfb_offsets,
            num_sfb,
            &scalefactors,
            &mut self.channel.noise_seed,
            &mut spectral,
        );

        let mut out = vec![0.0f32; FRAME_LEN];
        self.channel.filterbank.synthesize(
            &spectral,
            WindowSequence::OnlyLongSequence,
            WindowShape::Sine,
            WindowShape::Sine,
            &mut self.channel.overlap,
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

/// One FD channel-pair element's worth of decoder state.
pub struct UsacFdStereoDecoder {
    layout: Layout,
    channels: [ChannelState; 2],
}

impl UsacFdStereoDecoder {
    pub fn new() -> Self {
        Self { layout: Layout::new(), channels: [ChannelState::new(), ChannelState::new()] }
    }

    /// Decode one stereo frame's raw block into a `(left, right)` pair of
    /// `FRAME_LEN` PCM sample vectors.
    pub fn decode_frame(&mut self, reader: &mut BitReader) -> Result<(Vec<f32>, Vec<f32>)> {
        let num_sfb = self.layout.num_sfb;
        let mut ms_mask = vec![false; num_sfb];
        for used in ms_mask.iter_mut() {
            *used = reader.read_bit()?;
        }

        let mut spectral = [vec![0.0f32; FRAME_LEN], vec![0.0f32; FRAME_LEN]];
        for (ch, spec) in self.channels.iter_mut().zip(spectral.iter_mut()) {
            let scalefactors = read_scale_factor_data(reader, num_sfb)?;
            let pairs = FRAME_LEN / 2;
            let mut quant = vec![0i32; FRAME_LEN];
            decode_pairs(reader, &mut ch.contexts, pairs, pairs, &mut quant);
            dequantize_channel(
                &quant,
                &self.layout.sfb_offsets,
                num_sfb,
                &scalefactors,
                &mut ch.noise_seed,
                spec,
            );
        }

        // Undo mid/side in place wherever the encoder used it: the
        // transmitted pair `(m, s)` reconstructs as `(m + s, m - s)`, the
        // same inverse `apply_ms_stereo` uses for AAC-LC
        // (`src/decoder/aac/stereo.rs`).
        let [mid, side] = &mut spectral;
        for (b, &used) in ms_mask.iter().enumerate().take(num_sfb) {
            if !used {
                continue;
            }
            let lo = self.layout.sfb_offsets[b];
            let hi = self.layout.sfb_offsets[b + 1].min(FRAME_LEN);
            for i in lo..hi {
                let m = mid[i];
                let s = side[i];
                mid[i] = m + s;
                side[i] = m - s;
            }
        }

        let mut out = [vec![0.0f32; FRAME_LEN], vec![0.0f32; FRAME_LEN]];
        for ((ch, spec), o) in self.channels.iter_mut().zip(spectral.iter()).zip(out.iter_mut()) {
            ch.filterbank.synthesize(
                spec,
                WindowSequence::OnlyLongSequence,
                WindowShape::Sine,
                WindowShape::Sine,
                &mut ch.overlap,
                o,
            );
        }

        let [left, right] = out;
        Ok((left, right))
    }
}

impl Default for UsacFdStereoDecoder {
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
