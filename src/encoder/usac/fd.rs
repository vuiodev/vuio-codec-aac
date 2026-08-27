//! Minimal USAC frequency-domain (FD) single-channel-element encoder.
//!
//! This is the smallest real slice of ISO/IEC 23003-3's `UsacFrame()` that
//! exercises the spectral arithmetic coder in
//! [`crate::encoder::usac::arith`] end to end: one channel, FD core mode
//! only, fixed 1024-sample long windows. What it deliberately does not
//! cover — ACELP, TCX, stereo channel pairs, TNS, noise filling, MPEG
//! Surround, and the general `UsacConfig()`/element-tree signalling a real
//! stream needs — is separate, larger work; see the arithmetic-coder
//! module's own doc for why that split makes sense.
//!
//! # Reused, not reinvented
//!
//! Two things this frame needs turn out to already exist, verified against
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
//! * **The 3/4-power quantization law.** `ixheaacd_esc_iquant`'s fixed-point
//!   reconstruction (`magnitude^(4/3) * 2^((scalefactor-100)/4)`, see
//!   [`crate::decoder::usac::fd`]'s doc) is the same law AAC-LC's
//!   [`crate::encoder::aac::quant`] already implements in floating point, so
//!   quantization here calls straight into `quant::quantize_band` rather
//!   than a second implementation of the same curve.
//!
//! # One scalefactor per frame
//!
//! A real encoder varies the scalefactor per band to shape quantization
//! noise perceptually; that is a rate-distortion problem this minimal path
//! does not attempt. Instead every band shares one scalefactor chosen from
//! the whole frame's peak, and `scale_factor_data()` still carries one
//! Huffman-coded delta per band (all zero after the first) — a real,
//! spec-shaped bitstream, just not an adaptive one. The scalefactor is
//! floored at [`SF_OFFSET`] (100): the reference decoder's fixed-point
//! reconstruction hard-zeros any band whose `scalefactor - 100` goes
//! negative (`ixheaacd_apply_scfs_and_nf`'s `if (fac < 0) fac_fix = 0`), so
//! a real encoder never emits one below that floor.

use crate::bitstream::BitWriter;
use crate::dsp::fft::Complex32;
use crate::dsp::mdct::MdctContext;
use crate::dsp::window::generate_sine_window_f32;
use crate::encoder::aac::huffman::write_scalefactor_delta;
use crate::encoder::aac::quant::{MAX_QUANT_MAGNITUDE, SF_OFFSET, initial_scalefactor, quantize_band};
use crate::encoder::usac::arith::encode_pairs;
use crate::tables::scalefactor::{MAX_SFB_LONG, compute_sfb_offsets};
use crate::tables::sfb::SFB_48_1024;
use crate::tables::usac_arith::Contexts;

/// Samples per frame this minimal path codes. USAC's other standard core
/// frame length (768) and every short-window path are out of scope.
pub const FRAME_LEN: usize = 1024;

/// One FD single-channel element's worth of encoder state.
pub struct UsacFdEncoder {
    mdct: MdctContext,
    window: Vec<f32>,
    /// Previous frame's samples, forming this frame's window's first half.
    history: Vec<f32>,
    spectrum: Vec<f32>,
    scratch: Vec<Complex32>,
    /// Bands `scale_factor_data()` carries. One scalefactor covers the whole
    /// spectrum here (see module docs), but the syntax still spells out one
    /// delta per band, so the encoder and decoder have to agree on the count.
    num_sfb: usize,
    /// Arithmetic-coder context history, carried across frames.
    contexts: Contexts,
}

impl UsacFdEncoder {
    pub fn new() -> Self {
        let mut offsets = [0usize; MAX_SFB_LONG + 1];
        let count = compute_sfb_offsets(SFB_48_1024, &mut offsets);
        let mdct = MdctContext::new(FRAME_LEN);
        let scratch_len = mdct.scratch_len();
        Self {
            mdct,
            window: generate_sine_window_f32(2 * FRAME_LEN),
            history: vec![0.0; FRAME_LEN],
            spectrum: vec![0.0; FRAME_LEN],
            scratch: vec![Complex32::default(); scratch_len],
            num_sfb: count - 1,
            contexts: Contexts::new(),
        }
    }

    /// Encode one 1024-sample frame into a byte-aligned raw block:
    /// `global_gain`, `scale_factor_data()`, then the arithmetic-coded
    /// spectral data.
    pub fn encode_frame(&mut self, pcm: &[f32]) -> Vec<u8> {
        assert_eq!(pcm.len(), FRAME_LEN);

        let mut windowed = vec![0.0f32; 2 * FRAME_LEN];
        for i in 0..FRAME_LEN {
            windowed[i] = self.history[i] * self.window[i];
            windowed[FRAME_LEN + i] = pcm[i] * self.window[FRAME_LEN + i];
        }
        self.mdct.forward(&windowed, &mut self.spectrum, &mut self.scratch);
        self.history.copy_from_slice(pcm);

        // Aim the frame's peak at three quarters of the codebook's ceiling,
        // leaving headroom so a slightly louder next frame does not have to
        // escape-code every coefficient.
        let target_peak = MAX_QUANT_MAGNITUDE as f32 * 0.75;
        let scalefactor = initial_scalefactor(&self.spectrum, target_peak).max(SF_OFFSET);

        let mut quant = vec![0i32; FRAME_LEN];
        quantize_band(&self.spectrum, scalefactor, &mut quant);

        let mut writer = BitWriter::with_capacity(FRAME_LEN);
        writer.write_u8(scalefactor.clamp(0, 255) as u8, 8);
        for _ in 1..self.num_sfb {
            write_scalefactor_delta(&mut writer, 0);
        }

        let pairs = FRAME_LEN / 2;
        encode_pairs(&mut writer, &mut self.contexts, &quant, pairs, pairs);

        writer.byte_align_zero();
        writer.into_bytes()
    }
}

impl Default for UsacFdEncoder {
    fn default() -> Self {
        Self::new()
    }
}
