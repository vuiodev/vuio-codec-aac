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
//! # Noise filling
//!
//! `FDChannelStream()` also carries `noise_level` (3 bits) and
//! `noise_offset` (5 bits) right after `global_gain` — the reference only
//! sends these when the stream's `noiseFilling` config flag is set, but this
//! minimal path has no persistent per-element config, so it always carries
//! the field and lets `noise_level == 0` mean "off this frame" (a valid,
//! spec-legal value, not a special case). Reproduced from
//! `ixheaacd_apply_scfs_and_nf`: a coefficient only gets synthesized rather
//! than left silent when its *band's start* offset is at or past
//! [`NOISE_FILLING_START_OFFSET`] (noise filling never reaches into the
//! bands a real encoder actually spends bits shaping), and a band that
//! quantized to *all* zeros has its effective scale shifted by
//! `noise_offset - 16` first, since a fully-silent band's transmitted
//! scalefactor says nothing about what level of noise belongs there.
//!
//! `noise_offset - 16` being a no-op shift at `noise_offset == 16` is not a
//! coincidence this decoder relies on — see
//! [`crate::encoder::usac::fd`]'s noise-filling docs for why the encoder
//! here always transmits exactly that value.
//!
//! # Temporal noise shaping
//!
//! Inverted after dequantization but *before* undoing mid/side (for the
//! stereo path) — the exact reverse of the encoder's order, which applies
//! TNS after the mid/side decision has already turned the two spectra into
//! mid/side. See [`crate::encoder::usac::tns`] for the filter itself and why
//! its coefficient tables and step-up recursion are shared with classic
//! AAC-LC's TNS; this decoder reuses [`crate::decoder::aac::tns::ar_filter`]
//! directly rather than a second synthesis-filter implementation.
//!
//! `tns_data_present`, the coefficient-resolution bit, `length` and `order`
//! are read back in the exact order `crate::encoder::usac::fd`'s
//! `write_tns_data` writes them; `length` is parsed and discarded rather than
//! acted on, since [`crate::encoder::usac::tns::filter_band_range`] gives
//! both sides the same answer without either transmitting it.
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
//! per-band array from [`read_channel_header`] and dequantizes each
//! band's coefficient range with its own value, matching what
//! [`crate::encoder::usac::fd`]'s real per-band rate loop now transmits.

use crate::bitstream::BitReader;
use crate::decoder::aac::huffman::decode_scalefactor_delta;
use crate::decoder::aac::ics::{MAX_TNS_ORDER, SF_OFFSET};
use crate::decoder::aac::tns::{TNS_PARCOR_4, ar_filter, parcor_to_lpc};
use crate::decoder::usac::arith::{decode_pairs, dequantize};
use crate::dsp::filterbank::Filterbank;
use crate::encoder::usac::tns::{self, TnsFilter};
use crate::error::Result;
use crate::tables::scalefactor::{MAX_SFB_LONG, compute_sfb_offsets};
use crate::tables::sfb::SFB_48_1024;
use crate::tables::usac_arith::{Contexts, POW_14_3, TABLE_EXP, TABLE_FRAC};
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

/// Coefficient index below which noise filling never applies, matching
/// `ixheaacd_apply_scfs_and_nf`'s `1024`-sample-frame, long-window case
/// (`(usac_data->ccfl == 768) ? ... : (islong ? 160 : 20)` — this minimal
/// path is always the `1024`/long branch). A band only counts once its
/// *start* reaches this offset, not merely any coefficient inside it — a
/// band straddling the boundary is left alone entirely, same as the
/// reference.
pub const NOISE_FILLING_START_OFFSET: usize = 160;

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

/// One channel's `FDChannelStream()` header: `global_gain`, the noise-filling
/// side info, and the per-band scalefactor array — everything
/// [`dequantize_channel`] needs, read in the exact order
/// [`crate::encoder::usac::fd::write_channel_header`] writes it.
struct ChannelHeader {
    noise_level: i32,
    noise_offset: i32,
    scalefactors: Vec<i32>,
    tns: Option<TnsFilter>,
}

/// Read one channel's header: `global_gain` (8 bits), `noise_level` (3
/// bits), `noise_offset` (5 bits), one Huffman-coded scalefactor delta per
/// remaining band, then the TNS side info.
fn read_channel_header(reader: &mut BitReader, num_sfb: usize) -> Result<ChannelHeader> {
    let mut scalefactors = vec![0i32; num_sfb];
    scalefactors[0] = reader.read_u8(8)? as i32;
    let noise_level = reader.read_u8(3)? as i32;
    let noise_offset = reader.read_u8(5)? as i32;
    for b in 1..num_sfb {
        scalefactors[b] = scalefactors[b - 1] + decode_scalefactor_delta(reader)?;
    }
    let tns = read_tns_data(reader)?;
    Ok(ChannelHeader { noise_level, noise_offset, scalefactors, tns })
}

/// Read `tns_data_present` (1 bit) and, when set, the filter's side info —
/// the exact inverse of `crate::encoder::usac::fd`'s `write_tns_data`. The
/// coefficient-resolution bit and `length` field are consumed but not acted
/// on: this path only ever transmits 4-bit-resolution coefficients, and the
/// band range the filter covers is a fixed function of `num_sfb` (see
/// [`tns::filter_band_range`]), not something either side needs to read back.
fn read_tns_data(reader: &mut BitReader) -> Result<Option<TnsFilter>> {
    if !reader.read_bit()? {
        return Ok(None);
    }
    let _coef_res_bit = reader.read_bit()?;
    let _length = reader.read_u8(6)?;
    let order = reader.read_u8(4)? as usize;

    let mut coef = [0i8; MAX_TNS_ORDER];
    if order > 0 {
        let _direction = reader.read_bit()?;
        let _coef_compress = reader.read_bit()?;
        let width = tns::COEF_RES_BITS as usize;
        let shift = 32 - width;
        for c in coef.iter_mut().take(order) {
            let raw = reader.read_u32(width)? as i32;
            // Sign-extend from `width` bits, the same trick AAC-LC's own
            // TNS parser uses for its own (differently-sized) coefficients.
            *c = ((raw << shift) >> shift) as i8;
        }
    }
    Ok(Some(TnsFilter { order, coef }))
}

/// Undo TNS on one channel's dequantized spectrum in place: resolve the
/// transmitted reflection-coefficient indices back through
/// [`TNS_PARCOR_4`] (the reference decoder's `ixheaacd_tns_dec_coef_usac`
/// indexes the same table the same way), convert to direct-form LPC, and run
/// the all-pole synthesis filter over the same band range
/// [`crate::encoder::usac::tns::apply`] filtered — a no-op when no filter was
/// transmitted.
fn invert_tns(spectral: &mut [f32], sfb_offsets: &[usize], num_sfb: usize, filter: &TnsFilter) {
    if filter.order == 0 {
        return;
    }
    let bias = (TNS_PARCOR_4.len() / 2) as i32;
    let mut quantized = [0.0f32; MAX_TNS_ORDER];
    for i in 0..filter.order {
        quantized[i] = TNS_PARCOR_4[(filter.coef[i] as i32 + bias) as usize];
    }
    let mut lpc = [0.0f32; MAX_TNS_ORDER + 1];
    parcor_to_lpc(&quantized[..filter.order], &mut lpc);

    let (start_band, stop_band) = tns::filter_band_range(num_sfb);
    let lo = sfb_offsets[start_band];
    let hi = sfb_offsets[stop_band].min(spectral.len());
    if hi > lo {
        ar_filter(&mut spectral[lo..hi], &lpc, filter.order, false);
    }
}

/// Dequantize one channel's coded magnitudes band by band, each with its own
/// scalefactor, synthesizing noise wherever [`NOISE_FILLING_START_OFFSET`]
/// and a nonzero `noise_level` say to (see module docs), and scale the
/// arithmetic coder's fixed-point output down into the same linear units the
/// encoder quantized in.
fn dequantize_channel(
    quant: &[i32],
    sfb_offsets: &[usize],
    num_sfb: usize,
    header: &ChannelHeader,
    noise_seed: &mut u32,
    spectral: &mut [f32],
) {
    let noise_level_fixed = POW_14_3[header.noise_level.clamp(0, 7) as usize];
    let mut coef = vec![0i32; quant.len()];
    for b in 0..num_sfb {
        let lo = sfb_offsets[b];
        let hi = sfb_offsets[b + 1].min(quant.len());
        if lo >= hi {
            continue;
        }
        let present = header.noise_level != 0 && lo >= NOISE_FILLING_START_OFFSET;
        let band_all_zero = quant[lo..hi].iter().all(|&q| q == 0);
        let scalefactor = if present && band_all_zero {
            header.scalefactors[b] + (header.noise_offset - 16)
        } else {
            header.scalefactors[b]
        };
        dequantize(
            &quant[lo..hi],
            &mut coef[lo..hi],
            noise_level_fixed,
            present,
            noise_seed,
            fac_fix(scalefactor),
        );
    }
    for (s, &c) in spectral.iter_mut().zip(coef.iter()) {
        *s = c as f32 / ARITH_FIXED_POINT_SHIFT;
    }
    if let Some(filter) = &header.tns {
        invert_tns(spectral, sfb_offsets, num_sfb, filter);
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
        let header = read_channel_header(reader, num_sfb)?;

        let pairs = FRAME_LEN / 2;
        let mut quant = vec![0i32; FRAME_LEN];
        decode_pairs(reader, &mut self.channel.contexts, pairs, pairs, &mut quant);

        let mut spectral = vec![0.0f32; FRAME_LEN];
        dequantize_channel(
            &quant,
            &self.layout.sfb_offsets,
            num_sfb,
            &header,
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
            let header = read_channel_header(reader, num_sfb)?;
            let pairs = FRAME_LEN / 2;
            let mut quant = vec![0i32; FRAME_LEN];
            decode_pairs(reader, &mut ch.contexts, pairs, pairs, &mut quant);
            dequantize_channel(
                &quant,
                &self.layout.sfb_offsets,
                num_sfb,
                &header,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> Layout {
        Layout::new()
    }

    /// A band whose quantized values are all zero, at or past
    /// [`NOISE_FILLING_START_OFFSET`], must come out of dequantization
    /// nonzero when noise filling is on: this is the whole point of the
    /// tool, and it is easy to wire up a gate that never actually fires.
    #[test]
    fn noise_filling_synthesizes_a_fully_zeroed_high_band() {
        let layout = layout();
        let num_sfb = layout.num_sfb;
        let sfb = num_sfb - 1;
        let lo = layout.sfb_offsets[sfb];
        let hi = layout.sfb_offsets[sfb + 1].min(FRAME_LEN);
        assert!(lo >= NOISE_FILLING_START_OFFSET, "test needs a band past the start offset");

        let mut quant = vec![0i32; FRAME_LEN];
        // Give a low band real (nonzero) content so it is not itself
        // eligible for the whole-band shift, and leave the high test band
        // at its default all-zero.
        quant[0] = 5;

        let mut scalefactors = vec![SF_OFFSET; num_sfb];
        scalefactors[sfb] = SF_OFFSET + 40;
        let header = ChannelHeader { noise_level: 5, noise_offset: 16, scalefactors, tns: None };

        let mut spectral = vec![0.0f32; FRAME_LEN];
        let mut seed = 7u32;
        dequantize_channel(&quant, &layout.sfb_offsets, num_sfb, &header, &mut seed, &mut spectral);

        assert!(
            spectral[lo..hi].iter().any(|&v| v != 0.0),
            "a noise-filled band must not stay silent"
        );
    }

    /// `noise_level == 0` must leave every zeroed band exactly silent — the
    /// field's "off" value has to actually turn the tool off, not just
    /// pick the quietest nonzero level.
    #[test]
    fn noise_level_zero_leaves_zeroed_bands_silent() {
        let layout = layout();
        let num_sfb = layout.num_sfb;
        let quant = vec![0i32; FRAME_LEN];
        let scalefactors = vec![SF_OFFSET; num_sfb];
        let header = ChannelHeader { noise_level: 0, noise_offset: 0, scalefactors, tns: None };

        let mut spectral = vec![0.0f32; FRAME_LEN];
        let mut seed = 7u32;
        dequantize_channel(&quant, &layout.sfb_offsets, num_sfb, &header, &mut seed, &mut spectral);

        assert!(spectral.iter().all(|&v| v == 0.0), "noise_level 0 must mean no synthesis at all");
    }

    /// A band whose *start* sits before [`NOISE_FILLING_START_OFFSET`] must
    /// stay silent even with noise filling on — the reference gates on the
    /// band's start, not on any individual coefficient's position, so the
    /// last band entirely below the threshold is skipped entirely (as would
    /// be a band straddling it, though this table's band boundaries happen
    /// to land exactly on 160, so there is no such band to test directly).
    #[test]
    fn a_band_starting_before_the_offset_is_never_filled() {
        let layout = layout();
        let num_sfb = layout.num_sfb;
        let start_sfb = (0..num_sfb)
            .find(|&sfb| layout.sfb_offsets[sfb] >= NOISE_FILLING_START_OFFSET)
            .expect("some band must reach the start offset");
        let low_sfb = start_sfb - 1;
        let lo = layout.sfb_offsets[low_sfb];
        let hi = layout.sfb_offsets[low_sfb + 1].min(FRAME_LEN);
        assert!(lo < NOISE_FILLING_START_OFFSET, "test needs a band entirely below the offset");

        let quant = vec![0i32; FRAME_LEN];
        let mut scalefactors = vec![SF_OFFSET; num_sfb];
        scalefactors[low_sfb] = SF_OFFSET + 40;
        let header = ChannelHeader { noise_level: 7, noise_offset: 16, scalefactors, tns: None };

        let mut spectral = vec![0.0f32; FRAME_LEN];
        let mut seed = 7u32;
        dequantize_channel(&quant, &layout.sfb_offsets, num_sfb, &header, &mut seed, &mut spectral);

        assert!(
            spectral[lo..hi].iter().all(|&v| v == 0.0),
            "a band starting below the offset must never be noise-filled"
        );
    }
}
