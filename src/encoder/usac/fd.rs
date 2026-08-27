//! Minimal USAC frequency-domain (FD) encoder: single-channel and stereo.
//!
//! This is the smallest real slice of ISO/IEC 23003-3's `UsacFrame()` that
//! exercises the spectral arithmetic coder in
//! [`crate::encoder::usac::arith`] end to end: FD core mode only, fixed
//! 1024-sample long windows. What it deliberately does not cover — ACELP,
//! TCX, USAC's complex-prediction stereo, MPEG Surround, and the general
//! `UsacConfig()`/element-tree signalling a real stream needs — is separate,
//! larger work; see the arithmetic-coder module's own doc for why that split
//! makes sense.
//!
//! # Temporal noise shaping
//!
//! Applied to the spectrum before the rate loop measures or quantizes
//! anything, exactly where AAC-LC's own encoder runs it (see
//! `src/encoder/engine.rs`'s identical ordering) — it is the shaped residual
//! that has to get quantized and judged, not the raw spectrum. See
//! [`crate::encoder::usac::tns`] for the filter itself; this module only
//! decides where in the frame's header its side info sits.
//!
//! `tns_data_present` (1 bit) is transmitted explicitly rather than relying
//! on a persistent per-element config flag the way the reference does — the
//! same choice, and for the same reason, this module's noise-filling side
//! info already makes (see that section below).
//!
//! # Reused, not reinvented
//!
//! Three things this frame needs turn out to already exist, verified against
//! `c/libxaac`:
//!
//! * **Scalefactor coding.** `iusace_write_scf_data` in
//!   `c/libxaac/encoder/iusace_write_bitstream.c` codes scalefactor deltas
//!   through `iusace_huffman_code_table`, a 121-entry table indexed by
//!   `delta + 60` — the same shape as classic AAC-LC's scalefactor Huffman
//!   table, and spot-checking entries (e.g. index 60, delta 0, is the
//!   1-bit codeword `0` in both) confirms it *is* the same table. This
//!   encoder writes scalefactor deltas with
//!   [`crate::encoder::aac::huffman::write_scalefactor_delta`] rather than
//!   re-deriving the table.
//! * **The scalefactor band table.** `iusace_sfb_48_1024` in
//!   `c/libxaac/encoder/iusace_psy_rom.c` (used for both 44.1 kHz and 48 kHz
//!   at a 1024-sample frame) is, band-width for band-width, the same table
//!   as [`crate::tables::sfb::SFB_48_1024`] — confirmed by expanding both to
//!   cumulative offsets and comparing them. Reused directly rather than
//!   transcribed a second time.
//! * **The masking model and mid/side decision.** AAC-LC's
//!   [`crate::encoder::aac::psycho::PsychoacousticModel`] is driven purely by
//!   a band table, sample rate and bitrate, with no AAC-LC-specific
//!   assumption baked in, so it applies unchanged to USAC's FD spectrum.
//!   The stereo decision below mirrors `decide_mid_side` in
//!   `src/encoder/engine.rs` band for band (per-band energy comparison,
//!   0.98 tie-break towards left/right) — small enough to duplicate rather
//!   than restructure a working, tested file to share it.
//!
//! # A real per-band rate loop, adapted for an entropy coder with no table
//!
//! [`crate::encoder::aac::rate::RateLoop`] estimates a Huffman codebook's
//! cost from a table without writing anything. The arithmetic coder has no
//! such table — a symbol's cost depends on the adaptive model row its
//! context selects, which depends on every coefficient decoded before it in
//! the block — so [`fit_scalefactors`] measures each trial's true cost by
//! actually encoding it into a scratch buffer against a *cloned* copy of the
//! coder's context (never the real one: committing a trial would poison the
//! next block's context with a value that was never really sent). Only the
//! winning trial's quantization is kept; the real context only advances once,
//! in the final commit.
//!
//! # One decode-side bug this surfaced
//!
//! The previous single-scalefactor-per-frame version's decoder accumulated
//! every scalefactor delta into one final scalar and dequantized the whole
//! spectrum with it — harmless when every delta was zero, but wrong the
//! moment scalefactors actually vary per band. See
//! [`crate::decoder::usac::fd`] for the fix: the decoder now keeps the whole
//! per-band array and dequantizes each band with its own value.
//!
//! # Noise filling, simplified deliberately
//!
//! The reference encoder's `iusace_noise_filling` picks `noise_level` and
//! `noise_offset` from a log-domain ratio of "energy in bands that coded to
//! all zero" against "energy in bands that didn't", using several constants
//! (`alpha = 0.15`, and offsets of `-50`/`-58` in the two sides of the
//! ratio) that are calibrated to that encoder's own internal fixed-point
//! spectrum and scalefactor conventions — conventions this port does not
//! reuse (this encoder works in [`crate::encoder::aac::quant`]'s floating
//! point convention throughout). Transplanting those specific constants
//! without the matching internal scale would be porting numbers, not the
//! algorithm, and there is no reference decoder handy to check the result
//! against — so this chooses `noise_level`/`noise_offset` a different way,
//! one calibrated to this encoder's own units and checkable by construction:
//!
//! 1. [`crate::decoder::usac::fd::NOISE_FILLING_START_OFFSET`] and later
//!    bands that quantized to *all* zero are exactly the bands the decoder
//!    will noise-fill (see that module's docs) — so collect the true,
//!    pre-quantization spectral energy the encoder discarded in exactly
//!    those bands.
//! 2. Fix `noise_offset` at 16 — per the decoder's `scalefactor +
//!    (noise_offset - 16)` shift, this is a documented no-op, so a fully
//!    zeroed band's noise is reconstructed at exactly its own transmitted
//!    scalefactor's level, no extra parameter to solve for.
//! 3. For each of the 8 possible `noise_level` values, predict what the
//!    decoder will actually reconstruct (replaying
//!    `ixheaacd_esc_iquant`'s noise-branch arithmetic — the `>> 25`
//!    fixed-point shift, then this codec's `/ 64` unit conversion, both
//!    already established in [`crate::decoder::usac::fd`]) and keep whichever
//!    level's predicted magnitude is closest, in a least-squares sense, to
//!    the energy actually discarded. A discarded-energy floor near zero
//!    picks `noise_level = 0`, which both this encoder and the decoder treat
//!    as "off" — no separate flag needed.
//!
//! This does not reproduce the reference bit-for-bit, but the *bitstream
//! contract* — field widths, the start-offset rule, the whole-zeroed-band
//! shift — is exact, checked directly against `ixheaacd_apply_scfs_and_nf`;
//! what differs is only the perceptual judgment of which of the 8 legal
//! noise levels the encoder decides fits best.

use crate::bitstream::BitWriter;
use crate::decoder::usac::fd::NOISE_FILLING_START_OFFSET;
use crate::tables::usac_arith::{POW_14_3, TABLE_EXP, TABLE_FRAC};
use crate::dsp::fft::Complex32;
use crate::dsp::mdct::MdctContext;
use crate::dsp::window::generate_sine_window_f32;
use crate::encoder::aac::huffman::write_scalefactor_delta;
use crate::encoder::aac::psycho::{PsychoResult, PsychoacousticModel};
use crate::encoder::aac::quant::{MAX_QUANT_MAGNITUDE, SF_OFFSET, quantize_band};
use crate::encoder::aac::rate::{MAX_SCALEFACTOR, MAX_SCALEFACTOR_DELTA, MIN_SCALEFACTOR};
use crate::encoder::usac::arith::encode_pairs;
use crate::encoder::usac::tns::{self, TnsFilter, TnsSetup};
use crate::tables::scalefactor::{MAX_SFB_LONG, compute_sfb_offsets};
use crate::tables::sfb::SFB_48_1024;
use crate::tables::usac_arith::Contexts;
use crate::types::WindowSequence;

/// Samples per frame this minimal path codes. USAC's other standard core
/// frame length (768) and every short-window path are out of scope.
pub const FRAME_LEN: usize = 1024;

/// Sample rate the masking model is tuned for. [`SFB_48_1024`] is shared by
/// 44.1 kHz and 48 kHz per the reference table (see module docs), and the
/// model's own choice of sample rate only shifts the Bark-scale band
/// centres slightly within that shared table, so a fixed value here is a
/// reasonable stand-in for a minimal path that does not thread a real
/// sample rate through yet.
const MODEL_SAMPLE_RATE_HZ: u32 = 44_100;
/// Bitrate the masking model is tuned for; only selects which of its two
/// hole-tolerance presets applies, not a correctness-affecting choice.
const MODEL_BITRATE_BPS: u32 = 64_000;
/// Default payload bit budget for one channel's frame, generous enough that
/// a loud, dynamic test signal still exercises the arithmetic coder's
/// escape path (see `tests/usac_fd_round_trip.rs`) while still being a real
/// budget a caller can tighten with [`UsacFdEncoder::set_budget_bits`].
pub const DEFAULT_BUDGET_BITS: usize = 12_000;

/// Bands [`SFB_48_1024`] resolves to, plus the model built from it — shared
/// setup both the mono and stereo encoders need, so it is not duplicated in
/// each.
struct Layout {
    sfb_offsets: [usize; MAX_SFB_LONG + 1],
    num_sfb: usize,
    tns: TnsSetup,
}

impl Layout {
    fn new() -> Self {
        let mut sfb_offsets = [0usize; MAX_SFB_LONG + 1];
        let count = compute_sfb_offsets(SFB_48_1024, &mut sfb_offsets);
        let num_sfb = count - 1;
        let tns = TnsSetup::new(&sfb_offsets[..=num_sfb], num_sfb);
        Self { sfb_offsets, num_sfb, tns }
    }
}

/// One channel's worth of transform, masking-model and arithmetic-coder state.
struct ChannelState {
    history: Vec<f32>,
    spectrum: Vec<f32>,
    model: PsychoacousticModel,
    psycho: PsychoResult,
    contexts: Contexts,
}

impl ChannelState {
    fn new(sfb_offsets: &[usize]) -> Self {
        Self {
            history: vec![0.0; FRAME_LEN],
            spectrum: vec![0.0; FRAME_LEN],
            model: PsychoacousticModel::new(
                MODEL_SAMPLE_RATE_HZ,
                MODEL_BITRATE_BPS,
                sfb_offsets,
                false,
            ),
            psycho: PsychoResult::default(),
            contexts: Contexts::new(),
        }
    }

    /// Window this frame's samples against history and forward-MDCT them.
    fn transform(
        &mut self,
        pcm: &[f32],
        window: &[f32],
        mdct: &MdctContext,
        scratch: &mut [Complex32],
    ) {
        let mut windowed = vec![0.0f32; 2 * FRAME_LEN];
        for i in 0..FRAME_LEN {
            windowed[i] = self.history[i] * window[i];
            windowed[FRAME_LEN + i] = pcm[i] * window[FRAME_LEN + i];
        }
        mdct.forward(&windowed, &mut self.spectrum, scratch);
        self.history.copy_from_slice(pcm);
    }
}

/// One FD single-channel element's worth of encoder state.
pub struct UsacFdEncoder {
    mdct: MdctContext,
    window: Vec<f32>,
    scratch: Vec<Complex32>,
    layout: Layout,
    channel: ChannelState,
    budget_bits: usize,
}

impl UsacFdEncoder {
    pub fn new() -> Self {
        let mdct = MdctContext::new(FRAME_LEN);
        let scratch = vec![Complex32::default(); mdct.scratch_len()];
        let layout = Layout::new();
        let channel = ChannelState::new(&layout.sfb_offsets[..=layout.num_sfb]);
        Self { mdct, window: generate_sine_window_f32(2 * FRAME_LEN), scratch, channel, layout, budget_bits: DEFAULT_BUDGET_BITS }
    }

    /// Change the payload bit budget a frame's `fit_scalefactors` search
    /// targets. A tighter budget trades quantization noise for size, the
    /// same tradeoff [`crate::encoder::aac::rate::RateLoop`] makes for
    /// AAC-LC.
    pub fn set_budget_bits(&mut self, bits: usize) {
        self.budget_bits = bits;
    }

    /// Encode one 1024-sample frame into a byte-aligned raw block:
    /// `global_gain`, the noise-filling side info, `scale_factor_data()`,
    /// then the arithmetic-coded spectral data.
    pub fn encode_frame(&mut self, pcm: &[f32]) -> Vec<u8> {
        assert_eq!(pcm.len(), FRAME_LEN);
        self.channel.transform(pcm, &self.window, &self.mdct, &mut self.scratch);

        let num_sfb = self.layout.num_sfb;
        let sfb_offsets = &self.layout.sfb_offsets[..=num_sfb];
        let tns_filter =
            tns::apply(&mut self.channel.spectrum, sfb_offsets, &self.layout.tns);

        let mut scalefactors = vec![SF_OFFSET; num_sfb];
        let mut quant = vec![0i32; FRAME_LEN];
        fit_scalefactors(
            &self.channel.spectrum,
            sfb_offsets,
            num_sfb,
            &mut self.channel.model,
            &mut self.channel.psycho,
            &self.channel.contexts,
            self.budget_bits,
            &mut scalefactors,
            &mut quant,
        );
        let (noise_level, noise_offset) =
            choose_noise_level(&self.channel.spectrum, sfb_offsets, num_sfb, &scalefactors, &quant);

        let mut writer = BitWriter::with_capacity(FRAME_LEN);
        write_channel_header(
            &mut writer,
            &scalefactors,
            noise_level,
            noise_offset,
            tns_filter.as_ref(),
            &self.layout.tns,
            num_sfb,
        );
        let pairs = FRAME_LEN / 2;
        encode_pairs(&mut writer, &mut self.channel.contexts, &quant, pairs, pairs);
        writer.byte_align_zero();
        writer.into_bytes()
    }
}

impl Default for UsacFdEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// One FD channel-pair element's worth of encoder state: two channels coded
/// independently except for a per-band mid/side decision made on their
/// spectra before quantization.
pub struct UsacFdStereoEncoder {
    mdct: MdctContext,
    window: Vec<f32>,
    scratch: Vec<Complex32>,
    layout: Layout,
    channels: [ChannelState; 2],
    /// Whether band `b` is coded as mid/side rather than left/right.
    ///
    /// Unlike AAC-LC's `ms_mask_present`, this minimal path always
    /// transmits one bit per band rather than an optional "no bands use
    /// it" fast path — simpler to get right, at the cost of a small,
    /// constant amount of overhead per frame.
    ms_mask: Vec<bool>,
    budget_bits: usize,
}

impl UsacFdStereoEncoder {
    pub fn new() -> Self {
        let mdct = MdctContext::new(FRAME_LEN);
        let scratch = vec![Complex32::default(); mdct.scratch_len()];
        let layout = Layout::new();
        let offsets = &layout.sfb_offsets[..=layout.num_sfb];
        let channels = [ChannelState::new(offsets), ChannelState::new(offsets)];
        Self {
            mdct,
            window: generate_sine_window_f32(2 * FRAME_LEN),
            scratch,
            channels,
            ms_mask: vec![false; layout.num_sfb],
            layout,
            budget_bits: DEFAULT_BUDGET_BITS,
        }
    }

    /// Change the payload bit budget each channel's `fit_scalefactors`
    /// search targets (applied independently to both channels).
    pub fn set_budget_bits(&mut self, bits: usize) {
        self.budget_bits = bits;
    }

    /// Encode one 1024-sample stereo frame into a byte-aligned raw block:
    /// the per-band mid/side mask, then each channel's `global_gain` +
    /// noise-filling side info + `scale_factor_data()` + arithmetic-coded
    /// spectral data in turn.
    pub fn encode_frame(&mut self, left: &[f32], right: &[f32]) -> Vec<u8> {
        assert_eq!(left.len(), FRAME_LEN);
        assert_eq!(right.len(), FRAME_LEN);

        let [ch0, ch1] = &mut self.channels;
        ch0.transform(left, &self.window, &self.mdct, &mut self.scratch);
        ch1.transform(right, &self.window, &self.mdct, &mut self.scratch);

        self.decide_and_apply_ms();

        let mut writer = BitWriter::with_capacity(2 * FRAME_LEN);
        for &used in &self.ms_mask {
            writer.write_bit(used);
        }

        let num_sfb = self.layout.num_sfb;
        let sfb_offsets = &self.layout.sfb_offsets[..=num_sfb];
        for ch in &mut self.channels {
            let tns_filter = tns::apply(&mut ch.spectrum, sfb_offsets, &self.layout.tns);

            let mut scalefactors = vec![SF_OFFSET; num_sfb];
            let mut quant = vec![0i32; FRAME_LEN];
            fit_scalefactors(
                &ch.spectrum,
                sfb_offsets,
                num_sfb,
                &mut ch.model,
                &mut ch.psycho,
                &ch.contexts,
                self.budget_bits,
                &mut scalefactors,
                &mut quant,
            );
            let (noise_level, noise_offset) =
                choose_noise_level(&ch.spectrum, sfb_offsets, num_sfb, &scalefactors, &quant);
            write_channel_header(
                &mut writer,
                &scalefactors,
                noise_level,
                noise_offset,
                tns_filter.as_ref(),
                &self.layout.tns,
                num_sfb,
            );
            let pairs = FRAME_LEN / 2;
            encode_pairs(&mut writer, &mut ch.contexts, &quant, pairs, pairs);
        }

        writer.byte_align_zero();
        writer.into_bytes()
    }

    /// Compare each band's left/right vs. mid/side coded energy on the
    /// as-yet-unquantized spectra and apply mid/side in place wherever it
    /// wins — the same decision `decide_mid_side` makes in
    /// `src/encoder/engine.rs`, band for band.
    fn decide_and_apply_ms(&mut self) {
        let num_sfb = self.layout.num_sfb;
        let offsets = &self.layout.sfb_offsets;
        let [left, right] = &mut self.channels;

        for b in 0..num_sfb {
            let lo = offsets[b];
            let hi = offsets[b + 1].min(FRAME_LEN);
            let mut lr = 0.0f64;
            let mut ms = 0.0f64;
            for i in lo..hi {
                let l = left.spectrum[i] as f64;
                let r = right.spectrum[i] as f64;
                lr += l * l + r * r;
                let m = 0.5 * (l + r);
                let s = 0.5 * (l - r);
                ms += m * m + s * s;
            }
            // A tie goes to left/right, which never makes anything worse.
            self.ms_mask[b] = ms < lr * 0.98;
        }

        for b in 0..num_sfb {
            if !self.ms_mask[b] {
                continue;
            }
            let lo = offsets[b];
            let hi = offsets[b + 1].min(FRAME_LEN);
            for i in lo..hi {
                let l = left.spectrum[i];
                let r = right.spectrum[i];
                left.spectrum[i] = 0.5 * (l + r);
                right.spectrum[i] = 0.5 * (l - r);
            }
        }
    }
}

impl Default for UsacFdStereoEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Write `global_gain` followed by one Huffman-coded delta per remaining
/// band, exactly the shape [`crate::decoder::usac::fd`] reads back.
///
/// Used only for [`attempt`]'s trial-cost measurement, which has no noise
/// side info to add yet (that's chosen from the *final* winning trial, after
/// the search is done) — a fixed 8-bit difference from the real header
/// every trial shares equally, so it does not affect which trial wins. The
/// real output goes through [`write_channel_header`] instead.
fn write_scale_factor_data(writer: &mut BitWriter, scalefactors: &[i32]) {
    writer.write_u8(scalefactors[0].clamp(0, 255) as u8, 8);
    let mut previous = scalefactors[0];
    for &sf in &scalefactors[1..] {
        write_scalefactor_delta(writer, sf - previous);
        previous = sf;
    }
}

/// Write one channel's real `FDChannelStream()` header: `global_gain` (8
/// bits), `noise_level` (3 bits), `noise_offset` (5 bits), one Huffman-coded
/// scalefactor delta per remaining band, then the TNS side info — the exact
/// order [`crate::decoder::usac::fd`]'s `read_channel_header` reads back
/// (see this module's noise-filling and TNS docs for why each field sits
/// where it does).
#[allow(clippy::too_many_arguments)]
fn write_channel_header(
    writer: &mut BitWriter,
    scalefactors: &[i32],
    noise_level: i32,
    noise_offset: i32,
    tns_filter: Option<&TnsFilter>,
    tns_setup: &TnsSetup,
    num_sfb: usize,
) {
    writer.write_u8(scalefactors[0].clamp(0, 255) as u8, 8);
    writer.write_u8(noise_level as u8, 3);
    writer.write_u8(noise_offset as u8, 5);
    let mut previous = scalefactors[0];
    for &sf in &scalefactors[1..] {
        write_scalefactor_delta(writer, sf - previous);
        previous = sf;
    }
    write_tns_data(writer, tns_filter, tns_setup, num_sfb);
}

/// Write `tns_data_present` (1 bit) and, when set, the filter's side info:
/// a coefficient-resolution bit, the (purely informational — see
/// [`TnsSetup::length_field`]) `length` field, `order` (4 bits, `order` is
/// capped at [`tns::ORDER`] which fits exactly), and — only when `order` is
/// nonzero — `direction` and `coef_compress` (both always `0` on this
/// long-window-only path) followed by `order` raw 4-bit two's-complement
/// coefficient indices, the same encoding
/// [`crate::encoder::engine`]'s AAC-LC `write_tns_data` uses for its own
/// coefficients.
fn write_tns_data(writer: &mut BitWriter, filter: Option<&TnsFilter>, setup: &TnsSetup, num_sfb: usize) {
    let Some(filter) = filter else {
        writer.write_bit(false);
        return;
    };
    writer.write_bit(true);
    writer.write_bit(true); // coef_res_bit: coef_res(4) - DEF_TNS_RES_OFFSET(3) = 1
    writer.write_u8(setup.length_field(num_sfb), 6);
    writer.write_u8(filter.order as u8, 4);
    if filter.order > 0 {
        writer.write_bit(false); // direction
        writer.write_bit(false); // coef_compress
        let mask = (1u32 << tns::COEF_RES_BITS) - 1;
        for i in 0..filter.order {
            writer.write_u32(filter.coef[i] as u32 & mask, tns::COEF_RES_BITS as usize);
        }
    }
}

/// Mirrors [`crate::decoder::usac::fd`]'s private `fac_fix` exactly — needed
/// here only to predict, for a candidate noise level, what the decoder will
/// actually reconstruct (see this module's noise-filling docs).
fn fac_fix_local(scalefactor: i32) -> i64 {
    let fac = scalefactor - SF_OFFSET;
    if fac < 0 {
        return 0;
    }
    let exp = (fac >> 2).min(31) as usize;
    let frac = (fac & 3) as usize;
    (TABLE_FRAC[3 + frac] as i64 * TABLE_EXP[exp]) >> 15
}

/// The same fixed-point-to-linear conversion documented at
/// [`crate::decoder::usac::fd`]'s `ARITH_FIXED_POINT_SHIFT`; duplicated here
/// (as an `f64`, for the noise-level search's precision) rather than made
/// `pub` there, matching this pair of modules' existing convention of small,
/// deliberate duplication over cross-module coupling.
const ARITH_FIXED_POINT_SHIFT: f64 = 64.0;

/// What the decoder will reconstruct for one noise-filled coefficient in a
/// band at `scalefactor`, given `noise_level_fixed = POW_14_3[noise_level]`
/// — replaying `ixheaacd_esc_iquant`'s noise branch (`(fac_fix * level) >>
/// 25`) and then this codec's fixed-point-to-linear scale, so the result is
/// directly comparable to this encoder's own (linear, unquantized) spectrum
/// values.
fn predicted_noise_magnitude(scalefactor: i32, noise_level_fixed: i32) -> f64 {
    let fac = fac_fix_local(scalefactor) as f64;
    (fac * noise_level_fixed as f64 / (1i64 << 25) as f64) / ARITH_FIXED_POINT_SHIFT
}

/// Choose `(noise_level, noise_offset)` for one channel's frame — see this
/// module's noise-filling docs for the reasoning. `noise_offset` is always
/// either `0` (noise filling off) or `16` (the shift-free value); only
/// `noise_level` is actually searched.
fn choose_noise_level(
    spectrum: &[f32],
    sfb_offsets: &[usize],
    num_sfb: usize,
    scalefactors: &[i32],
    quant: &[i32],
) -> (i32, i32) {
    let mut zeroed_bands: Vec<(usize, usize, i32)> = Vec::new();
    let mut discarded_energy = 0.0f64;
    let mut discarded_count = 0usize;

    for b in 0..num_sfb {
        let lo = sfb_offsets[b];
        let hi = sfb_offsets[b + 1].min(spectrum.len());
        if lo >= hi || lo < NOISE_FILLING_START_OFFSET {
            continue;
        }
        if quant[lo..hi].iter().all(|&q| q == 0) {
            for &x in &spectrum[lo..hi] {
                discarded_energy += (x as f64).powi(2);
            }
            discarded_count += hi - lo;
            zeroed_bands.push((lo, hi, scalefactors[b]));
        }
    }

    if zeroed_bands.is_empty() || discarded_count == 0 {
        return (0, 0);
    }
    // A discarded floor this quiet is not worth eight bits of side info —
    // and the least-squares search below would land on level 0 anyway, so
    // this is purely a fast path.
    if (discarded_energy / discarded_count as f64).sqrt() < 1.0 {
        return (0, 0);
    }

    let mut best_level = 0i32;
    let mut best_error = f64::MAX;
    for level in 0..=7i32 {
        let noise_level_fixed = POW_14_3[level as usize];
        let mut error = 0.0f64;
        for &(lo, hi, sf) in &zeroed_bands {
            let predicted = predicted_noise_magnitude(sf, noise_level_fixed);
            for &x in &spectrum[lo..hi] {
                let d = predicted - (x as f64).abs();
                error += d * d;
            }
        }
        if error < best_error {
            best_error = error;
            best_level = level;
        }
    }

    if best_level == 0 { (0, 0) } else { (best_level, 16) }
}

/// Search for the coarsest-fitting-under-budget set of per-band
/// scalefactors, the same bracket-then-bisect shape
/// [`crate::encoder::aac::rate::RateLoop::fit`] uses for AAC-LC, adapted for
/// an entropy coder with no per-symbol cost table (see module docs for why
/// that means measuring cost by actually encoding each trial).
///
/// Leaves the winning scalefactors and quantized coefficients in
/// `scalefactors`/`quant` and returns the bits that trial actually cost.
#[allow(clippy::too_many_arguments)]
fn fit_scalefactors(
    spectrum: &[f32],
    sfb_offsets: &[usize],
    num_sfb: usize,
    model: &mut PsychoacousticModel,
    psycho: &mut PsychoResult,
    contexts: &Contexts,
    budget_bits: usize,
    scalefactors: &mut [i32],
    quant: &mut [i32],
) -> usize {
    model.analyse(spectrum, sfb_offsets, WindowSequence::OnlyLongSequence, psycho);

    const FLOOR: f32 = -80.0;
    const CEILING: f32 = 120.0;
    const TOLERANCE: f32 = 0.5;

    let mut low = FLOOR;
    let mut high = CEILING;
    // The coarsest scale is the fallback answer if even it does not fit
    // (the smallest frame this spectrum can produce is still the best
    // available one), and doubles as the initial `high` bound below.
    let coarsest_bits =
        attempt(spectrum, sfb_offsets, num_sfb, psycho, contexts, high, scalefactors, quant);

    let finest_bits =
        attempt(spectrum, sfb_offsets, num_sfb, psycho, contexts, low, scalefactors, quant);
    if finest_bits <= budget_bits {
        // Even the finest quantization fits; nothing to trade away. `quant`
        // and `scalefactors` already hold this trial's result.
        return finest_bits;
    }

    let mut best_bits = coarsest_bits;

    while high - low > TOLERANCE {
        let mid = 0.5 * (low + high);
        let bits = attempt(spectrum, sfb_offsets, num_sfb, psycho, contexts, mid, scalefactors, quant);
        if bits <= budget_bits {
            high = mid;
            best_bits = bits;
        } else {
            low = mid;
        }
    }

    attempt(spectrum, sfb_offsets, num_sfb, psycho, contexts, high, scalefactors, quant);
    best_bits
}

/// One trial: pick a scalefactor per band from `psycho`'s thresholds scaled
/// by `scale_db`, quantize the spectrum with them, and measure the exact
/// bits that would cost by encoding into a scratch buffer against a cloned
/// context — never the real one, since a trial that is not the final choice
/// must not be allowed to poison the next block's context.
#[allow(clippy::too_many_arguments)]
fn attempt(
    spectrum: &[f32],
    sfb_offsets: &[usize],
    num_sfb: usize,
    psycho: &PsychoResult,
    contexts: &Contexts,
    scale_db: f32,
    scalefactors: &mut [i32],
    quant: &mut [i32],
) -> usize {
    let scale = 10f32.powf(scale_db / 10.0);
    let mut lowest = i32::MAX;

    for b in 0..num_sfb {
        let lo = sfb_offsets[b];
        let hi = sfb_offsets[b + 1].min(spectrum.len());
        if lo >= hi {
            scalefactors[b] = SF_OFFSET;
            continue;
        }
        let band = &spectrum[lo..hi];
        let peak = band.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if peak <= 0.0 {
            scalefactors[b] = SF_OFFSET;
            continue;
        }
        let threshold = (psycho.threshold[b] * scale).max(f32::MIN_POSITIVE);
        scalefactors[b] = band_scalefactor(band, threshold, peak);
        lowest = lowest.min(scalefactors[b]);
    }

    // Every coded band's scalefactor has to sit within one delta's reach of
    // the lowest, or the scalefactor codebook cannot express the
    // difference (see `RateLoop::attempt`'s identical reasoning).
    let ceiling = if lowest == i32::MAX { MAX_SCALEFACTOR } else { lowest + MAX_SCALEFACTOR_DELTA };
    for b in 0..num_sfb {
        let lo = sfb_offsets[b];
        let hi = sfb_offsets[b + 1].min(spectrum.len());
        let sf = scalefactors[b].min(ceiling).clamp(MIN_SCALEFACTOR, MAX_SCALEFACTOR);
        scalefactors[b] = sf;
        if lo < hi {
            quantize_band(&spectrum[lo..hi], sf, &mut quant[lo..hi]);
        }
    }

    let mut scratch_writer = BitWriter::with_capacity(quant.len());
    write_scale_factor_data(&mut scratch_writer, scalefactors);
    let mut scratch_contexts = contexts.clone();
    let pairs = quant.len() / 2;
    encode_pairs(&mut scratch_writer, &mut scratch_contexts, quant, pairs, pairs);
    scratch_writer.bits_written()
}

/// The scalefactor the usual estimate suggests for a band, the same
/// derivation `RateLoop`'s private `initial_scalefactor` uses for AAC-LC
/// (duplicated rather than made `pub` there, to keep the two rate loops
/// independent — see module docs).
fn band_scalefactor(band: &[f32], threshold: f32, peak: f32) -> i32 {
    let mut form_factor = 0.0f32;
    for &x in band {
        form_factor += x.abs().sqrt();
    }
    if form_factor <= f32::MIN_POSITIVE {
        return SF_OFFSET;
    }
    let ratio = (6.75 * threshold) / form_factor;
    let estimate = SF_OFFSET + ((8.0 / 3.0) * ratio.log2()).floor() as i32;
    estimate.max(smallest_representable(peak)).clamp(MIN_SCALEFACTOR, MAX_SCALEFACTOR)
}

/// Smallest scalefactor that keeps every coefficient inside the coded
/// range, mirroring `RateLoop`'s private helper of the same name.
fn smallest_representable(peak: f32) -> i32 {
    if peak <= 0.0 {
        return MIN_SCALEFACTOR;
    }
    let headroom = (MAX_QUANT_MAGNITUDE as f32) - 0.5;
    let sf = (16.0 / 3.0) * (0.75 * peak.log2() - headroom.log2());
    SF_OFFSET + sf.ceil() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A spectrum whose energy is wildly different from one band to the
    /// next must not collapse back to the old placeholder's one-scalefactor-
    /// for-everything behaviour: this is what actually distinguishes a real
    /// per-band rate loop from a global one wearing a per-band bitstream
    /// shape.
    #[test]
    fn scalefactors_adapt_to_per_band_energy() {
        let layout = Layout::new();
        let num_sfb = layout.num_sfb;
        let offsets = &layout.sfb_offsets[..=num_sfb];

        let mut spectrum = vec![0.0f32; FRAME_LEN];
        // A loud low band and a quiet high band, several bands apart so the
        // masking model's spreading does not blur them into each other.
        spectrum[offsets[0]..offsets[1]].fill(3.0e7);
        let quiet_band = num_sfb - 1;
        let quiet_hi = offsets[quiet_band + 1].min(FRAME_LEN);
        spectrum[offsets[quiet_band]..quiet_hi].fill(10.0);

        let mut model =
            PsychoacousticModel::new(MODEL_SAMPLE_RATE_HZ, MODEL_BITRATE_BPS, offsets, false);
        let mut psycho = PsychoResult::default();
        let contexts = Contexts::new();
        let mut scalefactors = vec![SF_OFFSET; num_sfb];
        let mut quant = vec![0i32; FRAME_LEN];

        fit_scalefactors(
            &spectrum,
            offsets,
            num_sfb,
            &mut model,
            &mut psycho,
            &contexts,
            DEFAULT_BUDGET_BITS,
            &mut scalefactors,
            &mut quant,
        );

        assert_ne!(
            scalefactors[0], scalefactors[quiet_band],
            "a loud band and a quiet band must not land on the same scalefactor"
        );
    }

    /// A tighter bit budget must produce a real, measurable reduction in
    /// cost, not just accept whatever the finest quantization happens to
    /// cost.
    #[test]
    fn tighter_budget_reduces_cost() {
        let layout = Layout::new();
        let num_sfb = layout.num_sfb;
        let offsets = &layout.sfb_offsets[..=num_sfb];

        let spectrum: Vec<f32> =
            (0..FRAME_LEN).map(|i| 1.0e6 * ((i as f32) * 0.037).sin()).collect();

        let mut model =
            PsychoacousticModel::new(MODEL_SAMPLE_RATE_HZ, MODEL_BITRATE_BPS, offsets, false);
        let mut psycho = PsychoResult::default();
        let contexts = Contexts::new();
        let mut scalefactors = vec![SF_OFFSET; num_sfb];
        let mut quant = vec![0i32; FRAME_LEN];

        let generous = fit_scalefactors(
            &spectrum, offsets, num_sfb, &mut model, &mut psycho, &contexts, 20_000,
            &mut scalefactors, &mut quant,
        );
        let tight = fit_scalefactors(
            &spectrum, offsets, num_sfb, &mut model, &mut psycho, &contexts, 500,
            &mut scalefactors, &mut quant,
        );

        assert!(tight < generous, "tightening the budget cost more bits: {tight} vs {generous}");
    }

    /// A band past the noise-filling start offset that quantized to all
    /// zero, but carried real pre-quantization energy, must pick a nonzero
    /// noise level — otherwise the search silently never turns the tool on.
    #[test]
    fn discarded_high_frequency_energy_turns_on_noise_filling() {
        let layout = Layout::new();
        let num_sfb = layout.num_sfb;
        let sfb = num_sfb - 1;
        let lo = layout.sfb_offsets[sfb];
        let hi = layout.sfb_offsets[sfb + 1].min(FRAME_LEN);
        assert!(lo >= NOISE_FILLING_START_OFFSET, "test needs a band past the start offset");

        let mut spectrum = vec![0.0f32; FRAME_LEN];
        spectrum[lo..hi].fill(500.0);
        let quant = vec![0i32; FRAME_LEN];
        let mut scalefactors = vec![SF_OFFSET; num_sfb];
        scalefactors[sfb] = SF_OFFSET + 40;

        let (level, offset) =
            choose_noise_level(&spectrum, &layout.sfb_offsets, num_sfb, &scalefactors, &quant);
        assert!(level > 0, "real discarded energy must not be ignored");
        assert_eq!(offset, 16, "a nonzero level always pairs with the shift-free offset");
    }

    /// A band that is not actually all zero must never be treated as a
    /// noise-filling candidate, whatever its unquantized energy — the tool
    /// only ever fills gaps a real coefficient could not otherwise reach.
    #[test]
    fn a_band_with_any_nonzero_coefficient_is_never_a_noise_candidate() {
        let layout = Layout::new();
        let num_sfb = layout.num_sfb;
        let sfb = num_sfb - 1;
        let lo = layout.sfb_offsets[sfb];
        let hi = layout.sfb_offsets[sfb + 1].min(FRAME_LEN);

        let mut spectrum = vec![0.0f32; FRAME_LEN];
        spectrum[lo..hi].fill(500.0);
        let mut quant = vec![0i32; FRAME_LEN];
        quant[lo] = 3;
        let mut scalefactors = vec![SF_OFFSET; num_sfb];
        scalefactors[sfb] = SF_OFFSET + 40;

        let (level, offset) =
            choose_noise_level(&spectrum, &layout.sfb_offsets, num_sfb, &scalefactors, &quant);
        assert_eq!((level, offset), (0, 0));
    }

    /// A spectrum with nothing meaningfully discarded above the start
    /// offset must leave noise filling off — the search should not invent
    /// side info to spend on silence that was genuinely silence.
    #[test]
    fn near_silence_leaves_noise_filling_off() {
        let layout = Layout::new();
        let num_sfb = layout.num_sfb;
        let spectrum = vec![0.0f32; FRAME_LEN];
        let quant = vec![0i32; FRAME_LEN];
        let scalefactors = vec![SF_OFFSET; num_sfb];

        let (level, offset) =
            choose_noise_level(&spectrum, &layout.sfb_offsets, num_sfb, &scalefactors, &quant);
        assert_eq!((level, offset), (0, 0));
    }
}
