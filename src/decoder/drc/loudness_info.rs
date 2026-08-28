//! `loudnessInfo()` from MPEG-D uniDRC (ISO/IEC 23003-4), the loudness half of
//! the standard this crate's [`super`] module does not yet implement.
//!
//! Ported from `c/libxaac/decoder/drc_src/impd_drc_static_payload.c`
//! (`impd_parse_loudness_info`, `impd_parse_loudness_measure`,
//! `impd_dec_method_value`).
//!
//! # What this covers, and what it does not
//!
//! `loudnessInfo()` is one measurement record within a much larger
//! `uniDrcConfig()`: a set of program/anchor/range loudness measurements for
//! one combination of DRC set, EQ set and downmix. This module parses that one
//! record — [`LoudnessInfo::parse`] — faithfully, including
//! [`decode_method_value`]'s per-measurement-type value encodings, which are
//! not uniform (loudness range alone has three different linear segments over
//! its 8-bit field). It does **not** cover the surrounding
//! `loudnessInfoSet()` (album loudness, the extension block) or any of
//! `uniDrcConfig()`'s DRC instructions, gain sets or EQ — see `text/plan.txt`
//! phase 6 for the rest. Nothing calls this yet: uniDRC's `uniDrcConfig()` and
//! `uniDrcGain()` extension payloads are not parsed anywhere in the decode
//! path, so this is a real, tested building block without a caller so far,
//! the same position `syntax::latm`/`syntax::adif` were in before being wired
//! into [`crate::decoder::engine::Decoder`].

use crate::bitstream::BitReader;
use crate::error::{Error, FormatError, Result};

/// How a loudness measurement was derived (`method_def`, a 4-bit field).
/// Each variant's value has its own encoding — see [`decode_method_value`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodDefinition {
    UnknownOther,
    ProgramLoudness,
    AnchorLoudness,
    MaxOfLoudnessRange,
    MomentaryLoudnessMax,
    ShortTermLoudnessMax,
    LoudnessRange,
    MixingLevel,
    RoomType,
    ShortTermLoudness,
}

impl MethodDefinition {
    fn from_u8(v: u8) -> Result<Self> {
        Ok(match v {
            0 => Self::UnknownOther,
            1 => Self::ProgramLoudness,
            2 => Self::AnchorLoudness,
            3 => Self::MaxOfLoudnessRange,
            4 => Self::MomentaryLoudnessMax,
            5 => Self::ShortTermLoudnessMax,
            6 => Self::LoudnessRange,
            7 => Self::MixingLevel,
            8 => Self::RoomType,
            9 => Self::ShortTermLoudness,
            _ => {
                return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
                    "loudnessInfo(): unknown method_def {v}"
                ))));
            }
        })
    }
}

/// Which loudness measurement standard produced a value (`measurement_system`,
/// a 4-bit field). Reference values above [`MeasurementSystem::Bs1771_1`] are
/// reserved by the standard and rejected, matching
/// `impd_parse_loudness_measure`'s own range check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementSystem {
    UnknownOther,
    EbuR128,
    Bs1770_4,
    Bs1770_4PreProcessing,
    User,
    ExpertPanel,
    Bs1771_1,
}

impl MeasurementSystem {
    fn from_u8(v: u8) -> Result<Self> {
        Ok(match v {
            0 => Self::UnknownOther,
            1 => Self::EbuR128,
            2 => Self::Bs1770_4,
            3 => Self::Bs1770_4PreProcessing,
            4 => Self::User,
            5 => Self::ExpertPanel,
            6 => Self::Bs1771_1,
            _ => {
                return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
                    "loudnessInfo(): reserved measurement_system {v}"
                ))));
            }
        })
    }
}

/// Decode a measurement's value field, whose width and scale depend entirely
/// on `method_def` (`impd_dec_method_value`).
///
/// Three distinct shapes appear here, and each is a real design choice in the
/// standard, not an arbitrary encoding:
/// * Most loudness measurements share one 8-bit, quarter-LU-step linear scale
///   from -57.75 LU (`UnknownOther` through `ShortTermLoudnessMax`).
/// * [`MethodDefinition::LoudnessRange`] instead uses three linear segments of
///   different step sizes over the same 8 bits, coarser at the high end where
///   a wide loudness range does not need fine resolution.
/// * [`MethodDefinition::MixingLevel`] and [`MethodDefinition::RoomType`] are
///   not loudness values at all (an SPL and a room-type enum), so they get
///   their own narrow fields and offsets.
pub fn decode_method_value(reader: &mut BitReader, method_def: MethodDefinition) -> Result<f32> {
    use MethodDefinition::*;
    Ok(match method_def {
        UnknownOther | ProgramLoudness | AnchorLoudness | MaxOfLoudnessRange
        | MomentaryLoudnessMax | ShortTermLoudnessMax => {
            let raw = reader.read_u32(8)?;
            -57.75 + raw as f32 * 0.25
        }
        LoudnessRange => {
            let raw = reader.read_u32(8)?;
            match raw {
                0 => 0.0,
                1..=128 => raw as f32 * 0.25,
                129..=204 => 0.5 * raw as f32 - 32.0,
                _ => raw as f32 - 134.0,
            }
        }
        MixingLevel => reader.read_u32(5)? as f32 + 80.0,
        RoomType => reader.read_u32(2)? as f32,
        ShortTermLoudness => {
            let raw = reader.read_u32(8)?;
            -116.0 + raw as f32 * 0.5
        }
    })
}

/// One loudness measurement within a [`LoudnessInfo`] record.
#[derive(Debug, Clone, Copy)]
pub struct LoudnessMeasure {
    pub method_def: MethodDefinition,
    pub method_val: f32,
    pub measurement_system: MeasurementSystem,
    /// How much to trust the measurement (0..=3); parsed because the
    /// bitstream carries it, not otherwise used by this port, matching the
    /// reference's own "parsed but unused" comment on the same field.
    pub reliability: u8,
}

impl LoudnessMeasure {
    fn parse(reader: &mut BitReader) -> Result<Self> {
        let method_def = MethodDefinition::from_u8(reader.read_u8(4)?)?;
        let method_val = decode_method_value(reader, method_def)?;
        let temp = reader.read_u8(6)?;
        let measurement_system = MeasurementSystem::from_u8((temp >> 2) & 0xF)?;
        let reliability = temp & 0b11;
        Ok(Self { method_def, method_val, measurement_system, reliability })
    }
}

/// A `loudnessInfo()` record: the loudness picture for one combination of DRC
/// set, EQ set and downmix (`impd_parse_loudness_info`).
#[derive(Debug, Clone)]
pub struct LoudnessInfo {
    pub drc_set_id: u8,
    /// Only present from `loudnessInfo()` version 1 onward; 0 otherwise.
    pub eq_set_id: u8,
    pub downmix_id: u8,
    /// `20.0 - raw * 0.03125` dBFS, or `None` when the field is absent or
    /// transmitted as the reserved "not present" value 0.
    pub sample_peak_level: Option<f32>,
    /// Same encoding as `sample_peak_level`, plus the two fields below.
    pub true_peak_level: Option<f32>,
    /// Parsed but not otherwise used by this port, matching the reference.
    pub true_peak_level_measurement_system: u8,
    pub true_peak_level_reliability: u8,
    pub measurements: Vec<LoudnessMeasure>,
    /// True when any measurement in this record is an anchor-loudness one.
    pub anchor_loudness_present: bool,
    /// True when any measurement in this record used expert-panel judgement.
    pub expert_loudness_present: bool,
}

/// A raw peak-level field decodes to `None` at the sentinel value 0
/// (`sample_peak_level == 0` means "not present" per the reference, decoded
/// separately from the presence flag that precedes it) and to
/// `20.0 - raw * 0.03125` dBFS otherwise.
fn decode_peak_level(raw: u32) -> Option<f32> {
    if raw == 0 { None } else { Some(20.0 - raw as f32 * 0.03125) }
}

impl LoudnessInfo {
    /// Parse one record. `version` selects whether `eq_set_id` is present
    /// (added in `loudnessInfo()` version 1).
    pub fn parse(reader: &mut BitReader, version: u32) -> Result<Self> {
        let drc_set_id = reader.read_u8(6)?;
        let eq_set_id = if version >= 1 { reader.read_u8(6)? } else { 0 };

        let temp = reader.read_u8(8)?;
        let downmix_id = (temp >> 1) & 0x7F;
        let sample_peak_level_present = temp & 1 == 1;
        let sample_peak_level =
            if sample_peak_level_present { decode_peak_level(reader.read_u32(12)?) } else { None };

        let true_peak_level_present = reader.read_bit()?;
        let (true_peak_level, true_peak_level_measurement_system, true_peak_level_reliability) =
            if true_peak_level_present {
                let level = decode_peak_level(reader.read_u32(12)?);
                let temp = reader.read_u8(6)?;
                (level, (temp >> 2) & 0xF, temp & 0b11)
            } else {
                (None, 0, 0)
            };

        let measurement_count = reader.read_u8(4)?;
        let mut measurements = Vec::with_capacity(measurement_count as usize);
        let mut anchor_loudness_present = false;
        let mut expert_loudness_present = false;
        for _ in 0..measurement_count {
            let m = LoudnessMeasure::parse(reader)?;
            anchor_loudness_present |= m.method_def == MethodDefinition::AnchorLoudness;
            expert_loudness_present |= m.measurement_system == MeasurementSystem::ExpertPanel;
            measurements.push(m);
        }

        Ok(Self {
            drc_set_id,
            eq_set_id,
            downmix_id,
            sample_peak_level,
            true_peak_level,
            true_peak_level_measurement_system,
            true_peak_level_reliability,
            measurements,
            anchor_loudness_present,
            expert_loudness_present,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;

    /// Every non-`LoudnessRange` method shares one linear scale; pin its two
    /// documented endpoints plus a mid-range value.
    #[test]
    fn program_loudness_uses_the_shared_linear_scale() {
        let cases = [(0u32, -57.75f32), (231, -0.0), (255, 6.0)];
        for (raw, want) in cases {
            let mut w = BitWriter::new();
            w.write_bits(raw as u64, 8);
            let bytes = w.finalize().to_vec();
            let mut r = BitReader::new(&bytes);
            let got = decode_method_value(&mut r, MethodDefinition::ProgramLoudness).unwrap();
            assert!((got - want).abs() < 0.01, "raw {raw}: {got} vs {want}");
        }
    }

    /// LoudnessRange's three segments must each hit their documented formula,
    /// and the boundaries between segments must not be off by one.
    #[test]
    fn loudness_range_switches_formula_at_its_two_segment_boundaries() {
        let cases = [
            (0u32, 0.0f32),     // sentinel zero
            (1, 0.25),          // first segment
            (128, 32.0),        // last value of the first segment
            (129, 32.5),        // first value of the second segment
            (204, 70.0),        // last value of the second segment
            (205, 71.0),        // first value of the third segment
            (255, 121.0),       // top of the range
        ];
        for (raw, want) in cases {
            let mut w = BitWriter::new();
            w.write_bits(raw as u64, 8);
            let bytes = w.finalize().to_vec();
            let mut r = BitReader::new(&bytes);
            let got = decode_method_value(&mut r, MethodDefinition::LoudnessRange).unwrap();
            assert!((got - want).abs() < 0.01, "raw {raw}: {got} vs {want}");
        }
    }

    #[test]
    fn mixing_level_and_room_type_use_their_own_narrow_fields() {
        let mut w = BitWriter::new();
        w.write_bits(10, 5); // mixing level raw value
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        assert_eq!(decode_method_value(&mut r, MethodDefinition::MixingLevel).unwrap(), 90.0);

        let mut w = BitWriter::new();
        w.write_bits(2, 2);
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        assert_eq!(decode_method_value(&mut r, MethodDefinition::RoomType).unwrap(), 2.0);
    }

    /// An unknown method_def or a reserved measurement_system must be
    /// rejected, not silently misparsed -- both are exactly the kind of
    /// "keep reading garbage" failure mode this port refuses to have.
    #[test]
    fn unknown_or_reserved_enum_values_are_rejected() {
        let mut w = BitWriter::new();
        w.write_bits(0b1111, 4); // method_def = 15, not a defined value
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        assert!(r.read_u8(4).is_ok());
        let mut r2 = BitReader::new(&bytes);
        let raw = r2.read_u8(4).unwrap();
        assert!(MethodDefinition::from_u8(raw).is_err());

        assert!(MeasurementSystem::from_u8(11).is_err());
        assert!(MeasurementSystem::from_u8(6).is_ok());
    }

    /// A full record with one anchor-loudness, expert-panel measurement must
    /// round-trip through the real bit widths and set both derived flags.
    #[test]
    fn a_full_loudness_info_record_parses_and_sets_its_derived_flags() {
        let mut w = BitWriter::new();
        w.write_bits(5, 6); // drc_set_id
        w.write_bits(3, 6); // eq_set_id (version >= 1)
        w.write_bits((12u64 << 1) | 1, 8); // downmix_id=12, sample_peak_level_present=1
        w.write_bits(500, 12); // sample_peak_level raw
        w.write_bit(true); // true_peak_level_present
        w.write_bits(600, 12); // true_peak_level raw
        w.write_bits(0, 6); // true_peak measurement system/reliability
        w.write_bits(1, 4); // measurement_count = 1
        w.write_bits(2, 4); // method_def = AnchorLoudness
        w.write_bits(100, 8); // method_val raw (shared linear scale)
        w.write_bits(5 << 2, 6); // measurement_system = ExpertPanel, reliability=0
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);

        let info = LoudnessInfo::parse(&mut r, 1).unwrap();
        assert_eq!(info.drc_set_id, 5);
        assert_eq!(info.eq_set_id, 3);
        assert_eq!(info.downmix_id, 12);
        assert!(info.sample_peak_level.is_some());
        assert!(info.true_peak_level.is_some());
        assert_eq!(info.measurements.len(), 1);
        assert!(info.anchor_loudness_present);
        assert!(info.expert_loudness_present);
    }

    /// version 0 must not read an eq_set_id field at all -- a caller passing
    /// the wrong version here would silently misalign every field after it.
    #[test]
    fn version_zero_has_no_eq_set_id_field() {
        let mut w = BitWriter::new();
        w.write_bits(1, 6); // drc_set_id
        w.write_bits(0, 8); // downmix_id=0, sample_peak_level_present=0
        w.write_bit(false); // true_peak_level_present
        w.write_bits(0, 4); // measurement_count = 0
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);

        let info = LoudnessInfo::parse(&mut r, 0).unwrap();
        assert_eq!(info.eq_set_id, 0);
        assert_eq!(info.measurements.len(), 0);
    }
}
