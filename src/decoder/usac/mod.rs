//! MPEG-D USAC (Unified Speech and Audio Coding) Core Decoder
//!
//! Implements spectral arithmetic decoding, Algebraic Vector Quantization (AVQ),
//! Linear Prediction Domain (LPD: ACELP speech core, TCX transform core),
//! Frequency Domain (FD) mode, and Forward Aliasing Cancellation (FAC)
//! (ISO/IEC 23003-3).
//!
//! # What is real here today
//!
//! * [`arith`] — the context-adaptive arithmetic decoder for spectral data.
//! * [`fd`] — the Frequency Domain core, end to end.
//! * [`acelp`] — the ACELP speech core: bitstream parse, both codebooks, gain
//!   VQ, post-processing and LPC synthesis.
//! * [`lsf`] — LSF dequantization, and via [`crate::tables::usac_lsf`] the
//!   LSF/LSP/LPC conversions an LPD frame needs to turn its transmitted
//!   envelope into a usable filter.
//!
//! TCX (the transform half of LPD) and FAC (which cancels the aliasing at a
//! core-mode switch) are not implemented yet — see `text/plan.txt` phase 1.7.
//! [`UsacDecoder::decode_lpd_frame`] therefore rejects a frame whose mode field
//! asks for TCX rather than returning something plausible-looking, following
//! this port's rule that an unsupported tool is an error and never a wrong
//! answer.

use crate::bitstream::BitReader;
use crate::error::{Error, FormatError, Result};
use crate::tables::usac_acelp::{LEN_SUBFR, ORDER};
use crate::tables::usac_lsf::{LPC_ORDER, LSF_INIT, lsf_to_lsp, lsp_to_lpc};

pub mod acelp;
pub mod arith;
pub mod avq;
pub mod container;
pub mod fd;
pub mod lsf;
pub mod tcx;

use acelp::{AcelpDecoder, AcelpFrame};

/// Core coding mode for a USAC audio frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsacCoreMode {
    /// Frequency Domain: an MDCT core, for music and general audio.
    FdMode,
    /// Linear Prediction Domain: ACELP or TCX, for speech.
    LpdMode,
}

/// What one subframe of an LPD superframe is coded with.
///
/// The 5-bit `lpd_mode` field packs the four subframes' choices into a single
/// number; [`LpdModeSet::parse`] unpacks it. TCX comes in three lengths because
/// a longer transform buys coding efficiency on stationary passages, where
/// ACELP's 64-sample granularity would spend bits re-describing the same
/// spectrum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpdSubframeMode {
    Acelp,
    Tcx20,
    Tcx40,
    Tcx80,
}

impl LpdSubframeMode {
    fn from_code(code: u8) -> Self {
        match code {
            0 => Self::Acelp,
            1 => Self::Tcx20,
            2 => Self::Tcx40,
            _ => Self::Tcx80,
        }
    }
}

/// The per-subframe mode assignment an LPD superframe carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LpdModeSet(pub [LpdSubframeMode; 4]);

impl LpdModeSet {
    /// Unpack the transmitted `lpd_mode` field (`ixheaacd_lpd_channel_stream`).
    ///
    /// The encoding is a run-length-ish ladder rather than four independent
    /// fields: 25 means one TCX80 across the whole superframe, 24 means two
    /// TCX40s, 20..24 and 16..20 mix a TCX40 half with two ACELP/TCX20 slots,
    /// and anything below 16 is four independent one-bit choices.
    pub fn parse(lpd_mode: u8) -> Result<Self> {
        if lpd_mode > 25 {
            return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
                "lpd_mode {lpd_mode} is out of range 0..=25"
            ))));
        }
        let m = match lpd_mode {
            25 => [3, 3, 3, 3],
            24 => [2, 2, 2, 2],
            20..=23 => [lpd_mode & 1, (lpd_mode >> 1) & 1, 2, 2],
            16..=19 => [2, 2, lpd_mode & 1, (lpd_mode >> 1) & 1],
            _ => [lpd_mode & 1, (lpd_mode >> 1) & 1, (lpd_mode >> 2) & 1, (lpd_mode >> 3) & 1],
        };
        Ok(Self(m.map(LpdSubframeMode::from_code)))
    }

    /// True when every subframe is ACELP, which is the only case this decoder
    /// can currently reconstruct without FAC.
    pub fn is_all_acelp(&self) -> bool {
        self.0.iter().all(|m| *m == LpdSubframeMode::Acelp)
    }
}

/// USAC Core Decoder engine: the LPD half.
///
/// The FD half lives in [`fd::UsacFdDecoder`]; this type owns the speech core
/// and the LSF history that lets one frame's envelope be interpolated from the
/// last one's.
pub struct UsacDecoder {
    acelp: AcelpDecoder,
    /// The previous frame's LSF set, which the current frame interpolates
    /// against so the synthesis filter moves smoothly across the boundary
    /// rather than stepping.
    prev_lsf: [f32; LPC_ORDER],
}

impl UsacDecoder {
    /// Create a decoder for a core running at the unscaled 12.8 kHz pitch grid.
    pub fn new() -> Self {
        Self::with_fscale(crate::tables::usac_acelp::FSCALE_DENOM)
    }

    /// Create a decoder whose pitch grid is scaled for a non-default core rate.
    pub fn with_fscale(fscale: i32) -> Self {
        Self { acelp: AcelpDecoder::new(fscale), prev_lsf: LSF_INIT }
    }

    /// Drop all history, as at a seek.
    pub fn reset(&mut self) {
        self.acelp.reset();
        self.prev_lsf = LSF_INIT;
    }

    /// Interpolate the four subframes' LPC filters between the previous frame's
    /// LSF set and this one's.
    ///
    /// Interpolation happens in the LSF domain, not on the LPC coefficients:
    /// LSFs stay ordered under a convex combination, and an ordered LSF set
    /// always converts back to a stable filter, which is exactly the guarantee
    /// interpolating raw LPC coefficients would not give.
    fn interpolate_lpc(&self, target: &[f32; LPC_ORDER]) -> [[f32; ORDER + 1]; 4] {
        let weights = [0.125f32, 0.375, 0.625, 0.875];
        weights.map(|w| {
            let mut lsf = [0.0f32; LPC_ORDER];
            for (i, slot) in lsf.iter_mut().enumerate() {
                *slot = (1.0 - w) * self.prev_lsf[i] + w * target[i];
            }
            lsp_to_lpc(&lsf_to_lsp(&lsf))
        })
    }

    /// Decode one LPD frame that is coded entirely with ACELP.
    ///
    /// `lsf` is the frame's dequantized envelope (see [`lsf::dequantize_lsf_abs`]),
    /// `stability` the LSF-derived factor that scales gain smoothing, and
    /// `core_mode` the `acelp_core_mode` field that fixes the innovation's bit
    /// budget. `out` receives `4 * LEN_SUBFR` samples.
    pub fn decode_acelp_frame(
        &mut self,
        frame: &AcelpFrame,
        lsf: &[f32; LPC_ORDER],
        stability: f32,
        core_mode: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let lpc = self.interpolate_lpc(lsf);
        self.acelp.decode_frame(frame, &lpc, stability, core_mode, out)?;
        self.prev_lsf = *lsf;
        Ok(())
    }

    /// Read and decode one LPD frame from a bitstream.
    ///
    /// Returns `Err` when the frame's `lpd_mode` selects TCX for any subframe:
    /// TCX and the FAC that bridges it are phase 1.7 work, and a partial answer
    /// here would be indistinguishable from a correct one at the API boundary.
    pub fn decode_lpd_frame(
        &mut self,
        reader: &mut BitReader,
        lsf: &[f32; LPC_ORDER],
        stability: f32,
        out: &mut [f32],
    ) -> Result<()> {
        let core_mode = reader.read_u32(3)? as usize;
        let modes = LpdModeSet::parse(reader.read_u32(5)? as u8)?;
        if !modes.is_all_acelp() {
            return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
                "LPD frame uses {:?}; TCX and FAC are not implemented yet",
                modes.0
            ))));
        }
        let frame = AcelpFrame::parse(reader, core_mode, 4)?;
        self.decode_acelp_frame(&frame, lsf, stability, core_mode, out)
    }

    /// Samples one LPD frame produces.
    pub const fn frame_len() -> usize {
        4 * LEN_SUBFR
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
    use crate::decoder::usac::lsf::dequantize_lsf_abs;

    /// The mode ladder is easy to get subtly wrong, and a wrong unpacking
    /// silently mis-routes subframes rather than failing, so pin the corners.
    #[test]
    fn the_lpd_mode_ladder_unpacks_to_the_reference_assignments() {
        use LpdSubframeMode::*;
        assert_eq!(LpdModeSet::parse(25).unwrap().0, [Tcx80; 4]);
        assert_eq!(LpdModeSet::parse(24).unwrap().0, [Tcx40; 4]);
        assert_eq!(LpdModeSet::parse(0).unwrap().0, [Acelp; 4]);
        assert_eq!(LpdModeSet::parse(15).unwrap().0, [Tcx20; 4]);
        assert_eq!(LpdModeSet::parse(20).unwrap().0, [Acelp, Acelp, Tcx40, Tcx40]);
        assert_eq!(LpdModeSet::parse(23).unwrap().0, [Tcx20, Tcx20, Tcx40, Tcx40]);
        assert_eq!(LpdModeSet::parse(16).unwrap().0, [Tcx40, Tcx40, Acelp, Acelp]);
        assert!(LpdModeSet::parse(26).is_err());
        assert!(LpdModeSet::parse(0).unwrap().is_all_acelp());
        assert!(!LpdModeSet::parse(24).unwrap().is_all_acelp());
    }

    /// Interpolating in the LSF domain must yield a stable filter at every
    /// subframe. Stability is what the whole LSF representation exists to
    /// guarantee, so if this fails the conversion is wrong, not the input.
    #[test]
    fn interpolated_filters_stay_stable_across_the_frame() {
        let mut dec = UsacDecoder::new();
        dec.prev_lsf = dequantize_lsf_abs(7);
        let target = dequantize_lsf_abs(200);
        let filters = dec.interpolate_lpc(&target);

        // A stable all-pole filter cannot blow up: drive each one with an
        // impulse and check the response decays rather than diverging.
        for (i, a) in filters.iter().enumerate() {
            let mut x = vec![0.0f32; 512];
            x[0] = 1.0;
            let mut y = vec![0.0f32; 512];
            let mut mem = [0.0f32; ORDER];
            acelp::synthesis(a, &x, &mut y, &mut mem);
            let head: f32 = y[..64].iter().map(|v| v.abs()).sum();
            let tail: f32 = y[448..].iter().map(|v| v.abs()).sum();
            assert!(y.iter().all(|v| v.is_finite()), "subframe {i} filter diverged");
            assert!(tail < head, "subframe {i} filter is not decaying: {tail} vs {head}");
        }
    }

    /// End to end from a bitstream: an all-ACELP frame decodes to real,
    /// bounded, non-silent samples, and the reader lands exactly where the
    /// frame's field widths say it should.
    #[test]
    fn an_all_acelp_frame_decodes_from_its_bitstream() {
        // core_mode = 2 (000 -> 010), lpd_mode = 0 (all ACELP), then payload.
        let mut bytes = vec![0x40u8, 0x00];
        bytes.extend(std::iter::repeat_n(0x9Cu8, 62));
        let mut reader = BitReader::new(&bytes);

        let mut dec = UsacDecoder::new();
        let lsf = dequantize_lsf_abs(120);
        let mut out = vec![0.0f32; UsacDecoder::frame_len()];
        dec.decode_lpd_frame(&mut reader, &lsf, 0.8, &mut out).unwrap();

        assert!(out.iter().all(|x| x.is_finite()), "synthesis diverged");
        assert!(out.iter().any(|x| *x != 0.0), "a real frame must not decode to silence");
        assert_eq!(reader.bit_position(), 3 + 5 + 2 + (9 + 6 + 9 + 6) + 4 * (1 + 36 + 7));
    }

    /// A TCX frame must be refused outright rather than silently decoded as
    /// something else -- the whole point of not shipping a placeholder.
    #[test]
    fn a_tcx_frame_is_refused_rather_than_faked() {
        // lpd_mode = 25 -> TCX80 across the superframe.
        let bytes = [0b0001_1001u8, 0x00, 0x00, 0x00];
        let mut reader = BitReader::new(&bytes);
        let mut dec = UsacDecoder::new();
        let lsf = dequantize_lsf_abs(0);
        let mut out = vec![0.0f32; UsacDecoder::frame_len()];
        let err = dec.decode_lpd_frame(&mut reader, &lsf, 0.8, &mut out).unwrap_err();
        assert!(format!("{err}").contains("TCX"), "unexpected error: {err}");
    }

    /// Two identical frames in a row must not produce identical output: the
    /// second one predicts from the first one's excitation. If history were
    /// being dropped this is what would catch it.
    #[test]
    fn consecutive_frames_carry_history_forward() {
        let mut dec = UsacDecoder::new();
        let lsf = dequantize_lsf_abs(60);
        let frame = AcelpFrame {
            mean_energy: 3,
            subframes: (0..4)
                .map(|s| acelp::SubframeParams {
                    acb_index: if s == 0 || s == 2 { 180 } else { 32 },
                    ltp_filtering: true,
                    icb_index: [7, 11, 19, 23, 0, 0, 0, 0],
                    gain_index: 70,
                })
                .collect(),
        };

        let mut first = vec![0.0f32; UsacDecoder::frame_len()];
        let mut second = vec![0.0f32; UsacDecoder::frame_len()];
        dec.decode_acelp_frame(&frame, &lsf, 0.8, 2, &mut first).unwrap();
        dec.decode_acelp_frame(&frame, &lsf, 0.8, 2, &mut second).unwrap();

        assert!(second.iter().all(|x| x.is_finite()));
        assert_ne!(first, second, "the second frame ignored the first frame's excitation");
    }
}
