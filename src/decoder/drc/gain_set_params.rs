//! `gainSetParams()` from MPEG-D uniDRC (ISO/IEC 23003-4): one gain set's
//! band structure and which transmitted gain sequence feeds each band.
//!
//! Ported from `c/libxaac/decoder/drc_src/impd_drc_static_payload.c`
//! (`impd_parse_gain_set_params`). Assembles
//! [`super::gain_modifiers::DrcCharacteristic`] into the larger record it
//! belongs to — see that module and [`super::channel_layout`],
//! [`super::downmix_instructions`], [`super::loudness_info`] for the other
//! self-contained pieces of the `uniDrcConfig()` this crate does not parse
//! yet, and `text/plan.txt` phase 6.1 for what remains.
//!
//! # What a gain set actually describes
//!
//! A gain set splits the spectrum into one or more bands (by crossover
//! frequency or by subband index — [`BandSplit`]) and names, for each band,
//! which *gain sequence* (a separately transmitted time series of gain
//! values, decoded elsewhere — dynamic payload, phase 6.2) drives it and what
//! DRC characteristic curve that sequence's values are gains *along*. The one
//! genuinely cross-record piece of state here is `gain_seq_idx`: gain
//! sequences are numbered once, consecutively, across every gain set in a
//! whole `uniDrcConfig()`, not restarted per gain set, so
//! [`GainSetParams::parse`] takes it as `&mut u32` — the direct, idiomatic
//! equivalent of the reference's in/out pointer parameter, since it really is
//! shared mutable state across sibling calls rather than something this
//! record can own.

use crate::bitstream::BitReader;
use crate::error::{Error, FormatError, Result};

use super::gain_modifiers::{BAND_COUNT_MAX, DrcCharacteristic};

/// How a gain sequence's values are coded (`gain_coding_profile`), which
/// governs how the dynamic payload interpolates between transmitted points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GainCodingProfile {
    /// Smoothly interpolated between nodes.
    Regular,
    /// Nodes ramp to a target and hold, for a fade.
    Fading,
    /// Nodes clip/duck rather than interpolate (`GAIN_CODING_PROFILE_DUCKING`
    /// is the same numeric value as `_CLIPPING` in the reference).
    ClippingOrDucking,
    /// One constant gain for the whole gain set; implies exactly one band
    /// and skips the band-structure fields entirely.
    Constant,
}

impl GainCodingProfile {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Regular,
            1 => Self::Fading,
            2 => Self::ClippingOrDucking,
            _ => Self::Constant,
        }
    }
}

/// How bands beyond the first divide the spectrum (`drc_band_type`), each
/// with `band_count - 1` entries (the first band's extent is implicit -- it
/// runs from the previous band's edge, or from the spectrum's start).
#[derive(Debug, Clone)]
pub enum BandSplit {
    /// A table-indexed crossover frequency per boundary.
    CrossoverFrequency(Vec<u8>),
    /// An explicit starting subband index per boundary.
    SubbandIndex(Vec<u16>),
}

/// One band's gain sequence assignment and DRC characteristic.
#[derive(Debug, Clone, Copy)]
pub struct BandGainParams {
    /// Index into the whole `uniDrcConfig()`'s gain sequences, unique per
    /// band across every gain set (see this module's docs).
    pub gain_seq_idx: u32,
    pub characteristic: DrcCharacteristic,
}

/// A parsed `gainSetParams()` record.
#[derive(Debug, Clone)]
pub struct GainSetParams {
    pub gain_coding_profile: GainCodingProfile,
    /// `true` selects linear interpolation between nodes rather than the
    /// alternative the standard defines.
    pub gain_interpolation_type: bool,
    pub full_frame: bool,
    pub time_alignment: bool,
    /// Minimum spacing between gain nodes, in some external time unit this
    /// module does not interpret; `None` when not transmitted (the decoder
    /// then uses a standard-defined default).
    pub time_delta_min: Option<u16>,
    pub bands: Vec<BandGainParams>,
    /// `None` when `band_count == 1` -- a single band has nothing to split.
    pub band_split: Option<BandSplit>,
}

/// Widest gain-sequence index a whole `uniDrcConfig()` may assign
/// (`SEQUENCE_COUNT_MAX`).
pub const SEQUENCE_COUNT_MAX: u32 = 24;

impl GainSetParams {
    /// Parse one record, advancing `*gain_seq_idx` by exactly as many gain
    /// sequences this gain set consumes (one per band, or one for the whole
    /// set under [`GainCodingProfile::Constant`]).
    pub fn parse(reader: &mut BitReader, version: u32, gain_seq_idx: &mut u32) -> Result<Self> {
        let temp = reader.read_u8(6)?;
        let gain_coding_profile = GainCodingProfile::from_u8((temp >> 4) & 3);
        let gain_interpolation_type = (temp >> 3) & 1 == 1;
        let full_frame = (temp >> 2) & 1 == 1;
        let time_alignment = (temp >> 1) & 1 == 1;
        let time_delt_min_flag = temp & 1 == 1;

        let time_delta_min = if time_delt_min_flag { Some(reader.read_u16(11)? + 1) } else { None };

        let (band_count, drc_band_type_is_crossover) = if gain_coding_profile == GainCodingProfile::Constant {
            (1u8, false)
        } else {
            let band_count = reader.read_u8(4)?;
            if band_count as usize > BAND_COUNT_MAX {
                return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
                    "gainSetParams(): band_count {band_count} exceeds {BAND_COUNT_MAX}"
                ))));
            }
            let drc_band_type = if band_count > 1 { reader.read_bit()? } else { false };
            (band_count, drc_band_type)
        };

        let mut bands = Vec::with_capacity(band_count as usize);
        for _ in 0..band_count {
            if version == 0 {
                *gain_seq_idx += 1;
            } else if reader.read_bit()? {
                *gain_seq_idx = reader.read_u32(6)?;
            } else {
                *gain_seq_idx += 1;
            }
            if *gain_seq_idx >= SEQUENCE_COUNT_MAX {
                return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
                    "gainSetParams(): gain_seq_idx {gain_seq_idx} exceeds {SEQUENCE_COUNT_MAX}"
                ))));
            }
            let characteristic = DrcCharacteristic::parse(reader, version)?;
            bands.push(BandGainParams { gain_seq_idx: *gain_seq_idx, characteristic });
        }

        let band_split = if band_count > 1 {
            Some(if drc_band_type_is_crossover {
                let mut v = Vec::with_capacity(band_count as usize - 1);
                for _ in 1..band_count {
                    v.push(reader.read_u8(4)?);
                }
                BandSplit::CrossoverFrequency(v)
            } else {
                let mut v = Vec::with_capacity(band_count as usize - 1);
                for _ in 1..band_count {
                    v.push(reader.read_u16(10)?);
                }
                BandSplit::SubbandIndex(v)
            })
        } else {
            None
        };

        Ok(Self {
            gain_coding_profile,
            gain_interpolation_type,
            full_frame,
            time_alignment,
            time_delta_min,
            bands,
            band_split,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;

    fn flags_byte(profile: u8, interp: bool, full_frame: bool, time_align: bool, delta_flag: bool) -> u8 {
        (profile << 4)
            | ((interp as u8) << 3)
            | ((full_frame as u8) << 2)
            | ((time_align as u8) << 1)
            | (delta_flag as u8)
    }

    /// A constant-profile gain set has exactly one band, no band_count field,
    /// no band-type bit, and no band split at all -- and at version 0, no
    /// per-band index-present bit either (that only exists from version 1),
    /// so the gain sequence index is simply advanced by one.
    #[test]
    fn constant_profile_implies_a_single_unsplit_band() {
        let mut w = BitWriter::new();
        w.write_bits(flags_byte(3, false, false, false, false) as u64, 6);
        w.write_bits(0, 7); // characteristic: version 0, index 0 = not present
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let mut idx = 5u32;
        let g = GainSetParams::parse(&mut r, 0, &mut idx).unwrap();

        assert_eq!(g.gain_coding_profile, GainCodingProfile::Constant);
        assert_eq!(g.bands.len(), 1);
        assert!(g.band_split.is_none());
        assert_eq!(g.bands[0].gain_seq_idx, 6);
        assert_eq!(idx, 6);
    }

    /// `time_delt_min_flag` set must decode `value + 1`, matching the
    /// reference's off-by-one (0 is never a valid minimum spacing).
    #[test]
    fn time_delta_min_is_decoded_plus_one() {
        let mut w = BitWriter::new();
        w.write_bits(flags_byte(3, false, false, false, true) as u64, 6);
        w.write_bits(9, 11); // -> 10
        w.write_bits(0, 7);
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let mut idx = 0u32;
        let g = GainSetParams::parse(&mut r, 0, &mut idx).unwrap();
        assert_eq!(g.time_delta_min, Some(10));
    }

    /// A multi-band, crossover-split gain set: band_count, the band-type bit,
    /// each band's own gain sequence + characteristic, then one crossover
    /// index per boundary (band_count - 1 of them).
    #[test]
    fn multi_band_crossover_split_reads_the_right_shape() {
        // Version 0's per-band shape carries no index-present bit (that only
        // exists from version 1) -- just each band's characteristic field.
        let mut w = BitWriter::new();
        w.write_bits(flags_byte(0, false, false, false, false) as u64, 6);
        w.write_bits(3, 4); // band_count = 3
        w.write_bit(true); // crossover split
        for _ in 0..3 {
            w.write_bits(0, 7); // characteristic not present, per band
        }
        for _ in 0..2 {
            w.write_bits(4, 4); // 2 crossover boundaries for 3 bands
        }
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let mut idx = 0u32;
        let g = GainSetParams::parse(&mut r, 0, &mut idx).unwrap();

        assert_eq!(g.bands.len(), 3);
        assert_eq!(g.bands.iter().map(|b| b.gain_seq_idx).collect::<Vec<_>>(), vec![1, 2, 3]);
        match g.band_split {
            Some(BandSplit::CrossoverFrequency(v)) => assert_eq!(v, vec![4, 4]),
            other => panic!("expected CrossoverFrequency, got {other:?}"),
        }
    }

    /// Version >= 1's explicit gain_seq_idx: when the index-present bit is
    /// set, the transmitted 6-bit value is used directly instead of
    /// auto-incrementing.
    #[test]
    fn version_1_can_assign_an_explicit_gain_seq_idx() {
        let mut w = BitWriter::new();
        w.write_bits(flags_byte(3, false, false, false, false) as u64, 6); // Constant
        w.write_bit(true); // index present
        w.write_bits(7, 6); // explicit index 7
        w.write_bit(false); // characteristic present = false (version >= 1 shape)
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let mut idx = 0u32;
        let g = GainSetParams::parse(&mut r, 1, &mut idx).unwrap();
        assert_eq!(g.bands[0].gain_seq_idx, 7);
        assert_eq!(idx, 7);
    }

    #[test]
    fn band_count_above_the_max_is_rejected() {
        let mut w = BitWriter::new();
        w.write_bits(flags_byte(0, false, false, false, false) as u64, 6);
        w.write_bits((BAND_COUNT_MAX + 1) as u64, 4);
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let mut idx = 0u32;
        assert!(GainSetParams::parse(&mut r, 0, &mut idx).is_err());
    }

    #[test]
    fn gain_seq_idx_overflowing_sequence_count_max_is_rejected() {
        let mut w = BitWriter::new();
        w.write_bits(flags_byte(3, false, false, false, false) as u64, 6);
        w.write_bits(0, 7);
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let mut idx = SEQUENCE_COUNT_MAX - 1; // one more push crosses the max
        assert!(GainSetParams::parse(&mut r, 0, &mut idx).is_err());
    }
}
