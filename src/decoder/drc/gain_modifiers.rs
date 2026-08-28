//! `gainSetParams()`'s two smaller pieces from MPEG-D uniDRC (ISO/IEC 23003-4):
//! which DRC characteristic curve a gain sequence follows
//! ([`DrcCharacteristic`], `impd_parse_gain_set_params_characteristics`), and
//! the per-band adjustments applied on top of it
//! ([`GainModifiers`], `impd_dec_gain_modifiers`).
//!
//! Ported from `c/libxaac/decoder/drc_src/impd_drc_static_payload.c`. Pairs
//! with [`super::channel_layout`], [`super::downmix_instructions`] and
//! [`super::loudness_info`] as further self-contained records inside the
//! larger `uniDrcConfig()` this crate does not parse yet — see those modules'
//! docs for the current state of uniDRC support and `text/plan.txt` phase 6.1
//! for what remains.
//!
//! # Why this doesn't reproduce the reference's uninitialized fields
//!
//! [`GainModifiers`] stores each per-band adjustment as an `Option`
//! ([`BandGainModifier::attn_ampl_scaling`], [`BandGainModifier::gain_offset`],
//! [`GainModifiers::shape_filter_idx`]) rather than a value alongside a
//! separate `_present`/`_flag` bit the way the reference's
//! `ia_gain_modifiers_struct` does. That is not a style preference: in the
//! reference, when a flag is clear the corresponding value field is simply
//! never written for that call, so a caller that reads it without checking
//! the flag first reads whatever was already in that memory -- a real
//! initialize-before-use hazard the struct's shape invites. Folding "present"
//! and "value" into one `Option` makes that mistake impossible to make here.

use crate::bitstream::BitReader;
use crate::error::{Error, FormatError, Result};

/// Widest band count a single `gainSetParams()` may carry (`BAND_COUNT_MAX`).
pub const BAND_COUNT_MAX: usize = 8;
/// Widest index a split (non-CICP) target characteristic may name
/// (`SPLIT_CHARACTERISTIC_COUNT_MAX`).
pub const SPLIT_CHARACTERISTIC_COUNT_MAX: u8 = 8;
/// Widest index a shape filter may name (`SHAPE_FILTER_COUNT_MAX`).
pub const SHAPE_FILTER_COUNT_MAX: u8 = 8;

/// Which DRC characteristic curve a gain sequence follows: either a single
/// CICP-defined index, or (version >= 1 only) a curve built by splitting two
/// separately-indexed halves.
#[derive(Debug, Clone, Copy, Default)]
pub struct DrcCharacteristic {
    /// `false` means no characteristic was signalled at all -- everything
    /// else on this type is then meaningless.
    pub present: bool,
    pub kind: CharacteristicKind,
}

#[derive(Debug, Clone, Copy)]
pub enum CharacteristicKind {
    /// A single CICP-defined characteristic index.
    Cicp(u8),
    /// A curve split into two independently indexed halves (version >= 1 only).
    Split { left_index: u8, right_index: u8 },
}

impl Default for CharacteristicKind {
    fn default() -> Self {
        Self::Cicp(0)
    }
}

impl DrcCharacteristic {
    /// Parse one `gainSetParams()` characteristic field
    /// (`impd_parse_gain_set_params_characteristics`).
    ///
    /// Version 0 only ever names a CICP index directly (`0` meaning "not
    /// present"); version 1 and up transmit an explicit presence bit, then a
    /// format bit choosing between a CICP index and a split curve.
    pub fn parse(reader: &mut BitReader, version: u32) -> Result<Self> {
        if version == 0 {
            let index = reader.read_u8(7)?;
            return Ok(if index > 0 {
                Self { present: true, kind: CharacteristicKind::Cicp(index) }
            } else {
                Self { present: false, kind: CharacteristicKind::default() }
            });
        }

        let present = reader.read_bit()?;
        if !present {
            return Ok(Self::default());
        }
        let format_is_cicp = reader.read_bit()?;
        let kind = if format_is_cicp {
            CharacteristicKind::Cicp(reader.read_u8(7)?)
        } else {
            let temp = reader.read_u8(8)?;
            CharacteristicKind::Split { left_index: (temp >> 4) & 0xf, right_index: temp & 0xf }
        };
        Ok(Self { present, kind })
    }
}

/// One band's adjustments on top of its DRC characteristic.
#[derive(Debug, Clone, Copy, Default)]
pub struct BandGainModifier {
    /// Override the gain sequence's characteristic for compression
    /// (attenuating) gains, by [`SPLIT_CHARACTERISTIC_COUNT_MAX`] index.
    /// Version >= 1 only -- always `None` under version 0.
    pub target_characteristic_left: Option<u8>,
    /// The same, for boosting (amplifying) gains.
    pub target_characteristic_right: Option<u8>,
    /// `(attenuation scale, amplification scale)`, each in `0.0..=1.875` in
    /// steps of `0.125` -- independent multipliers on the characteristic's
    /// negative and positive gain ranges respectively.
    pub attn_ampl_scaling: Option<(f32, f32)>,
    /// A constant offset added to every gain from this band, in dB, in
    /// `0.25` steps from `+-0.25` up to `+-8.0`.
    pub gain_offset: Option<f32>,
}

/// Per-band adjustments for one `gainSetParams()`, plus (single-band gain
/// sets only) a shared shape filter selection.
#[derive(Debug, Clone, Default)]
pub struct GainModifiers {
    pub bands: Vec<BandGainModifier>,
    /// Which shape filter to apply; only ever transmitted (and only ever
    /// `Some`) when this gain set has exactly one band and version >= 1.
    pub shape_filter_idx: Option<u8>,
}

/// Decode the shared "scaling byte" both version paths use: `(attn, ampl)`
/// each `nibble * 0.125`.
fn decode_scaling(reader: &mut BitReader) -> Result<(f32, f32)> {
    let temp = reader.read_u8(8)?;
    Ok((((temp >> 4) & 0xf) as f32 * 0.125, (temp & 0xf) as f32 * 0.125))
}

/// Decode the shared "offset field" both version paths use: 1 sign bit, 5
/// magnitude bits, `(1 + magnitude) * 0.25` dB.
fn decode_gain_offset(reader: &mut BitReader) -> Result<f32> {
    let temp = reader.read_u8(6)?;
    let magnitude = (1 + (temp & 0x1f)) as f32 * 0.25;
    Ok(if (temp >> 5) & 1 == 1 { -magnitude } else { magnitude })
}

impl GainModifiers {
    /// Parse `band_count` bands' worth of modifiers (`impd_dec_gain_modifiers`).
    ///
    /// Version 0 transmits one shared scaling and one shared offset for the
    /// whole gain set and (per the reference) copies them onto every band
    /// unchanged, with no per-band target-characteristic override and no
    /// shape filter at all; version >= 1 transmits everything per band, plus
    /// a shape filter when there is exactly one band.
    pub fn parse(reader: &mut BitReader, version: u32, band_count: usize) -> Result<Self> {
        if version == 0 {
            let attn_ampl_scaling =
                if reader.read_bit()? { Some(decode_scaling(reader)?) } else { None };
            let gain_offset = if reader.read_bit()? { Some(decode_gain_offset(reader)?) } else { None };
            let band = BandGainModifier {
                target_characteristic_left: None,
                target_characteristic_right: None,
                attn_ampl_scaling,
                gain_offset,
            };
            return Ok(Self { bands: vec![band; band_count], shape_filter_idx: None });
        }

        let mut bands = Vec::with_capacity(band_count);
        for _ in 0..band_count {
            let target_characteristic_left = if reader.read_bit()? {
                Some(read_split_characteristic_index(reader)?)
            } else {
                None
            };
            let target_characteristic_right = if reader.read_bit()? {
                Some(read_split_characteristic_index(reader)?)
            } else {
                None
            };
            let attn_ampl_scaling =
                if reader.read_bit()? { Some(decode_scaling(reader)?) } else { None };
            let gain_offset = if reader.read_bit()? { Some(decode_gain_offset(reader)?) } else { None };
            bands.push(BandGainModifier {
                target_characteristic_left,
                target_characteristic_right,
                attn_ampl_scaling,
                gain_offset,
            });
        }

        let shape_filter_idx = if band_count == 1 && reader.read_bit()? {
            let idx = reader.read_u8(4)?;
            if idx > SHAPE_FILTER_COUNT_MAX {
                return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
                    "gainSetParams(): shape_filter_idx {idx} exceeds {SHAPE_FILTER_COUNT_MAX}"
                ))));
            }
            Some(idx)
        } else {
            None
        };

        Ok(Self { bands, shape_filter_idx })
    }
}

fn read_split_characteristic_index(reader: &mut BitReader) -> Result<u8> {
    let idx = reader.read_u8(4)?;
    if idx >= SPLIT_CHARACTERISTIC_COUNT_MAX {
        return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
            "gainSetParams(): target characteristic index {idx} exceeds {SPLIT_CHARACTERISTIC_COUNT_MAX}"
        ))));
    }
    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;

    #[test]
    fn version_0_characteristic_zero_means_not_present() {
        let mut w = BitWriter::new();
        w.write_bits(0, 7);
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let c = DrcCharacteristic::parse(&mut r, 0).unwrap();
        assert!(!c.present);
    }

    #[test]
    fn version_0_nonzero_characteristic_is_a_cicp_index() {
        let mut w = BitWriter::new();
        w.write_bits(42, 7);
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let c = DrcCharacteristic::parse(&mut r, 0).unwrap();
        assert!(c.present);
        assert!(matches!(c.kind, CharacteristicKind::Cicp(42)));
    }

    #[test]
    fn version_1_split_characteristic_decodes_both_halves() {
        let mut w = BitWriter::new();
        w.write_bit(true); // present
        w.write_bit(false); // format_is_cicp = false -> split
        w.write_bits((5u64 << 4) | 9, 8); // left=5, right=9
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let c = DrcCharacteristic::parse(&mut r, 1).unwrap();
        assert!(c.present);
        match c.kind {
            CharacteristicKind::Split { left_index, right_index } => {
                assert_eq!((left_index, right_index), (5, 9));
            }
            other => panic!("expected Split, got {other:?}"),
        }
    }

    /// Version 0's shared scaling/offset must be copied onto every band
    /// unchanged, with no per-band override and no shape filter.
    #[test]
    fn version_0_broadcasts_one_scaling_and_offset_to_every_band() {
        let mut w = BitWriter::new();
        w.write_bit(true); // gain_scaling_flag
        w.write_bits((3u64 << 4) | 5, 8); // attn=3*0.125=0.375, ampl=5*0.125=0.625
        w.write_bit(true); // gain_offset_flag
        w.write_bits((1u64 << 5) | 3, 6); // sign=1 (negative), magnitude=(1+3)*0.25=1.0
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let g = GainModifiers::parse(&mut r, 0, 3).unwrap();

        assert_eq!(g.bands.len(), 3);
        for band in &g.bands {
            assert!((band.attn_ampl_scaling.unwrap().0 - 0.375).abs() < 1e-6);
            assert!((band.attn_ampl_scaling.unwrap().1 - 0.625).abs() < 1e-6);
            assert!((band.gain_offset.unwrap() - (-1.0)).abs() < 1e-6);
            assert!(band.target_characteristic_left.is_none());
        }
        assert!(g.shape_filter_idx.is_none());
    }

    /// Version >= 1 decodes each band independently, and only a single-band
    /// gain set may carry a shape filter.
    #[test]
    fn version_1_decodes_independent_bands_and_a_single_band_shape_filter() {
        let mut w = BitWriter::new();
        // Band 0: left characteristic present (index 2), right absent, no
        // scaling, no offset.
        w.write_bit(true);
        w.write_bits(2, 4);
        w.write_bit(false);
        w.write_bit(false); // gain_scaling_flag
        w.write_bit(false); // gain_offset_flag
        // shape_filter_flag (band_count == 1) + index
        w.write_bit(true);
        w.write_bits(3, 4);
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let g = GainModifiers::parse(&mut r, 1, 1).unwrap();

        assert_eq!(g.bands.len(), 1);
        assert_eq!(g.bands[0].target_characteristic_left, Some(2));
        assert!(g.bands[0].target_characteristic_right.is_none());
        assert!(g.bands[0].attn_ampl_scaling.is_none());
        assert!(g.bands[0].gain_offset.is_none());
        assert_eq!(g.shape_filter_idx, Some(3));
    }

    /// With more than one band, no shape filter bit is read at all -- even if
    /// one happened to be present in the buffer, it must not be consumed.
    #[test]
    fn multi_band_gain_sets_never_read_a_shape_filter_bit() {
        let mut w = BitWriter::new();
        for _ in 0..2 {
            w.write_bit(false); // left absent
            w.write_bit(false); // right absent
            w.write_bit(false); // no scaling
            w.write_bit(false); // no offset
        }
        w.write_bits(0xFF, 8); // trailing marker bits that must remain unread
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        let start = r.bit_position();
        let g = GainModifiers::parse(&mut r, 1, 2).unwrap();
        assert!(g.shape_filter_idx.is_none());
        assert_eq!(r.bit_position() - start, 8, "exactly 4 bits/band, no shape filter bit for band_count != 1");
    }

    #[test]
    fn out_of_range_indices_are_rejected() {
        let mut w = BitWriter::new();
        w.write_bit(true);
        w.write_bits(SPLIT_CHARACTERISTIC_COUNT_MAX as u64, 4); // == max, which is out of range (valid is < max)
        w.write_bit(false);
        w.write_bit(false);
        w.write_bit(false);
        let bytes = w.finalize().to_vec();
        let mut r = BitReader::new(&bytes);
        assert!(GainModifiers::parse(&mut r, 1, 1).is_err());
    }
}
