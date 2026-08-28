//! `downmixInstructions()` from MPEG-D uniDRC (ISO/IEC 23003-4): how to fold
//! [`super::channel_layout::ChannelLayout`]'s base channels down to a
//! narrower target layout, and at what per-pair gain.
//!
//! Ported from `c/libxaac/decoder/drc_src/impd_drc_static_payload.c`
//! (`impd_parse_dwnmix_instructions`). Pairs with [`super::channel_layout`]
//! and [`super::loudness_info`] as another self-contained record inside the
//! larger `uniDrcConfig()` this crate does not parse yet — see those modules'
//! docs for the current state of uniDRC support and `text/plan.txt` phase 6.1
//! for what remains.
//!
//! # Two incompatible coefficient encodings, by version
//!
//! `downmixInstructions()` version 0 sends one 4-bit code per (target
//! channel, base channel) pair, straight into [`DOWNMIX_COEFF_LFE`] or
//! [`DOWNMIX_COEFF`] depending on whether that base channel is LFE (an LFE
//! channel gets its own table because a sensible downmix range for a
//! subwoofer -- boosted as often as attenuated -- is nothing like a normal
//! channel's, which is attenuated or muted): each table entry is a dB value,
//! converted to a linear gain immediately.
//!
//! Version 1 replaces this with a single shared 32-entry table
//! ([`DOWNMIX_COEFF_V1`], 5 bits per pair, no separate LFE table) plus a
//! single transmitted *offset* applied to every coefficient afterward
//! ([`decode_offset`]) — chosen from four defined formulas by a 4-bit
//! selector, one of which (`3`) depends on the coefficients themselves (a
//! power-sum normalisation), which is why the offset can only be computed
//! after every raw coefficient is in hand, and the dB-to-linear conversion
//! has to wait until the offset is known too.

use crate::error::{Error, FormatError, Result};

/// Widest channel count either side of a downmix may have
/// (`MAX_CHANNEL_COUNT`, shared with [`super::channel_layout`]).
pub const MAX_CHANNEL_COUNT: u8 = super::channel_layout::MAX_CHANNEL_COUNT;

/// Version-0 per-pair dB table for a non-LFE base channel (`dwnmix_coeff`),
/// indexed by the transmitted 4-bit code. The last entry, -1000 dB, is
/// this table's way of encoding "muted" as an ordinary table lookup.
pub const DOWNMIX_COEFF: [f32; 16] = [
    0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, -6.0, -7.5, -9.0,
    -1000.0,
];

/// Version-0 per-pair dB table for an LFE base channel (`dwnmix_coeff_lfe`) --
/// centred near 0 dB with headroom to *boost* the subwoofer into a downmix
/// rather than only ever attenuating it, unlike [`DOWNMIX_COEFF`].
pub const DOWNMIX_COEFF_LFE: [f32; 16] = [
    10.0, 6.0, 4.5, 3.0, 1.5, 0.0, -1.5, -3.0, -4.5, -6.0, -10.0, -15.0, -20.0, -30.0, -40.0,
    -1000.0,
];

/// Version-1 shared per-pair dB table (`dwnmix_coeff_v1`), 5 bits, one table
/// for every base channel regardless of LFE.
pub const DOWNMIX_COEFF_V1: [f32; 32] = [
    10.0, 6.0, 4.5, 3.0, 1.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0,
    -5.5, -6.0, -6.5, -7.0, -7.5, -8.0, -9.0, -10.0, -11.0, -12.0, -15.0, -20.0, -25.0, -30.0,
    -40.0, -100_000.0,
];

/// A parsed `downmixInstructions()` record.
#[derive(Debug, Clone)]
pub struct DownmixInstructions {
    pub downmix_id: u8,
    pub target_channel_count: u8,
    /// CICP-style layout code for the target; this module does not interpret
    /// it, only carries it (same treatment [`super::channel_layout`] gives
    /// `defined_layout`).
    pub target_layout: u8,
    /// Per-(target channel, base channel) linear gain, row-major over target
    /// channels then base channels -- `target_channel_count *
    /// base_channel_count` entries when `downmix_coefficients_present`, empty
    /// otherwise (no coefficients were transmitted; a caller falls back to
    /// some default matrix of its own).
    pub downmix_coefficient: Vec<f32>,
}

/// Version >= 1's transmitted offset, applied (in dB) to every raw
/// coefficient before the dB-to-linear conversion. `raw_coefficients` is
/// needed only for selector `3`, which derives the offset from the
/// coefficients' own power sum rather than from the channel counts.
fn decode_offset(
    selector: u32,
    target_channel_count: u8,
    base_channel_count: u8,
    raw_coefficients: &[f32],
) -> Result<f32> {
    Ok(match selector {
        0 => 0.0,
        1 => {
            let a = 20.0 * (target_channel_count as f32 / base_channel_count as f32).log10();
            0.5 * (0.5 + a).floor()
        }
        2 => {
            let a = 20.0 * (target_channel_count as f32 / base_channel_count as f32).log10();
            0.5 * (0.5 + 2.0 * a).floor()
        }
        3 => {
            let sum: f32 = raw_coefficients.iter().map(|&db| 10f32.powf(0.1 * db)).sum();
            let b = 10.0 * sum.log10();
            0.5 * (0.5 + 2.0 * b).floor()
        }
        _ => {
            return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
                "downmixInstructions(): bs_dmix_offset {selector} is not one of the four defined values"
            ))));
        }
    })
}

impl DownmixInstructions {
    /// Parse one record. `base_channel_count` and `lfe_channel_map` come from
    /// this programme's [`super::channel_layout::ChannelLayout`] (`lfe_channel_map`
    /// need only be populated, and is only consulted, when `version == 0`).
    pub fn parse(
        reader: &mut crate::bitstream::BitReader,
        version: u32,
        base_channel_count: u8,
        lfe_channel_map: &[bool],
    ) -> Result<Self> {
        let temp = reader.read_u32(23)?;
        let downmix_id = ((temp >> 16) & 0x7f) as u8;
        let target_channel_count = ((temp >> 9) & 0x7f) as u8;
        if target_channel_count > MAX_CHANNEL_COUNT {
            return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
                "downmixInstructions(): target_channel_count {target_channel_count} exceeds {MAX_CHANNEL_COUNT}"
            ))));
        }
        let target_layout = ((temp >> 1) & 0xff) as u8;
        let downmix_coefficients_present = temp & 1 == 1;

        let mut downmix_coefficient = Vec::new();
        if downmix_coefficients_present {
            let pairs = target_channel_count as usize * base_channel_count as usize;
            if version == 0 {
                downmix_coefficient.reserve(pairs);
                for _ in 0..target_channel_count {
                    for base in 0..base_channel_count as usize {
                        let code = reader.read_u8(4)? as usize;
                        let db = if lfe_channel_map.get(base).copied().unwrap_or(false) {
                            DOWNMIX_COEFF_LFE[code]
                        } else {
                            DOWNMIX_COEFF[code]
                        };
                        downmix_coefficient.push(10f32.powf(0.05 * db));
                    }
                }
            } else {
                let bs_dmix_offset = reader.read_u32(4)?;
                let mut raw = Vec::with_capacity(pairs);
                for _ in 0..pairs {
                    let code = reader.read_u8(5)? as usize;
                    raw.push(DOWNMIX_COEFF_V1[code]);
                }
                let offset =
                    decode_offset(bs_dmix_offset, target_channel_count, base_channel_count, &raw)?;
                downmix_coefficient =
                    raw.into_iter().map(|db| 10f32.powf(0.05 * (db + offset))).collect();
            }
        }

        Ok(Self { downmix_id, target_channel_count, target_layout, downmix_coefficient })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::{BitReader, BitWriter};

    fn header(downmix_id: u8, target_channel_count: u8, target_layout: u8, coeffs_present: bool) -> BitWriter {
        let mut w = BitWriter::new();
        let temp = ((downmix_id as u32 & 0x7f) << 16)
            | ((target_channel_count as u32 & 0x7f) << 9)
            | ((target_layout as u32) << 1)
            | (coeffs_present as u32);
        w.write_bits(temp as u64, 23);
        w
    }

    /// Version 0's table split: an LFE base channel must read from
    /// `DOWNMIX_COEFF_LFE`, a normal one from `DOWNMIX_COEFF`, for the exact
    /// same transmitted code.
    #[test]
    fn version_0_splits_the_table_by_lfe_map() {
        let mut w = header(1, 1, 0, true);
        w.write_bits(5, 4); // code 5: DOWNMIX_COEFF[5]=-2.5, DOWNMIX_COEFF_LFE[5]=0.0
        let bytes = w.finalize().to_vec();

        let mut r = BitReader::new(&bytes);
        let non_lfe = DownmixInstructions::parse(&mut r, 0, 1, &[false]).unwrap();
        assert!((non_lfe.downmix_coefficient[0] - 10f32.powf(0.05 * -2.5)).abs() < 1e-6);

        let mut r = BitReader::new(&bytes);
        let lfe = DownmixInstructions::parse(&mut r, 0, 1, &[true]).unwrap();
        assert!((lfe.downmix_coefficient[0] - 1.0).abs() < 1e-6, "code 5 in the LFE table is 0 dB = unity gain");
    }

    /// Version 0's mute code (-1000 dB) must collapse to (numerically) zero
    /// linear gain, in both tables.
    #[test]
    fn version_0_mute_code_is_effectively_silent() {
        let mut w = header(0, 1, 0, true);
        w.write_bits(15, 4); // the last table entry in both tables
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let out = DownmixInstructions::parse(&mut r, 0, 1, &[false]).unwrap();
        assert!(out.downmix_coefficient[0] < 1e-40);
    }

    /// Selector 0 (no offset) must leave version-1 coefficients exactly at
    /// their table dB value converted straight to linear.
    #[test]
    fn version_1_selector_zero_applies_no_offset() {
        let mut w = header(2, 1, 0, true);
        w.write_bits(0, 4); // bs_dmix_offset = 0
        w.write_bits(5, 5); // DOWNMIX_COEFF_V1[5] = 0.0 dB
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let out = DownmixInstructions::parse(&mut r, 1, 1, &[false]).unwrap();
        assert!((out.downmix_coefficient[0] - 1.0).abs() < 1e-6);
    }

    /// An invalid `bs_dmix_offset` (only 0..=3 are defined) must be rejected,
    /// not silently treated as one of the valid selectors.
    #[test]
    fn version_1_invalid_offset_selector_is_rejected() {
        let mut w = header(0, 1, 0, true);
        w.write_bits(4, 4); // undefined selector
        w.write_bits(0, 5);
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        assert!(DownmixInstructions::parse(&mut r, 1, 1, &[false]).is_err());
    }

    /// No coefficients transmitted must leave the vector empty, not attempt
    /// to read bits that were never sent.
    #[test]
    fn absent_coefficients_leave_the_vector_empty_and_read_nothing_more() {
        let mut w = header(3, 2, 6, false);
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let out = DownmixInstructions::parse(&mut r, 0, 2, &[false, false]).unwrap();
        assert!(out.downmix_coefficient.is_empty());
        assert_eq!(out.downmix_id, 3);
        assert_eq!(out.target_channel_count, 2);
        assert_eq!(out.target_layout, 6);
    }

    /// `target_channel_count` beyond the shared max must be rejected before
    /// any coefficient bits are read.
    #[test]
    fn target_channel_count_above_the_max_is_rejected() {
        let mut w = header(0, MAX_CHANNEL_COUNT + 1, 0, false);
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        assert!(DownmixInstructions::parse(&mut r, 0, 2, &[false, false]).is_err());
    }

    /// A full target x base grid of coefficients must be read in row-major
    /// (target outer, base inner) order, matching the reference's nested loop.
    #[test]
    fn coefficients_are_read_in_target_major_base_minor_order() {
        let mut w = header(0, 2, 0, true);
        // target 0: base 0 -> code 0, base 1 -> code 1
        // target 1: base 0 -> code 2, base 1 -> code 3
        for code in [0u64, 1, 2, 3] {
            w.write_bits(code, 4);
        }
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let out = DownmixInstructions::parse(&mut r, 0, 2, &[false, false]).unwrap();
        assert_eq!(out.downmix_coefficient.len(), 4);
        let want: Vec<f32> =
            [0u8, 1, 2, 3].iter().map(|&c| 10f32.powf(0.05 * DOWNMIX_COEFF[c as usize])).collect();
        for (got, want) in out.downmix_coefficient.iter().zip(want.iter()) {
            assert!((got - want).abs() < 1e-6);
        }
    }
}
