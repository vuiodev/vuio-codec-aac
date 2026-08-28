//! `channelLayout()` from MPEG-D uniDRC (ISO/IEC 23003-4): which output
//! channels a `uniDrcConfig()` applies to, and which of them are LFE.
//!
//! Ported from `c/libxaac/decoder/drc_src/impd_drc_static_payload.c`
//! (`impd_parse_ch_layout`). Pairs with [`super::loudness_info`] as another
//! self-contained record inside the larger `uniDrcConfig()` this crate does
//! not parse yet — see that module's docs for the current state of uniDRC
//! support and `text/plan.txt` phase 6.1 for what remains.
//!
//! # Why LFE detection matters here
//!
//! A subwoofer channel needs no dynamic range control of its own (there is no
//! dialogue or transient content to protect), so uniDRC's gain sets can
//! exclude LFE channels rather than compressing them like every other
//! channel. Figuring out which channels those are is this record's whole
//! job: [`ChannelLayout::defined_layout`] either names a fixed, standard
//! loudspeaker layout (in which case which channels are LFE is implied and
//! not transmitted), or, when `defined_layout == 0`, each channel's speaker
//! position is sent explicitly and positions `3` and `26` (CICP speaker
//! position codes for the two defined LFE placements) mark that channel as
//! LFE — decoded here into [`ChannelLayout::lfe_channel_map`].
//!
//! # What the reference's cross-check becomes here
//!
//! The reference threads a shared `lfe_channel_map_count` through every
//! `channelLayout()` in a stream and rejects a record whose
//! `base_channel_count` disagrees with one already established. This module
//! has no shared parse-session state to thread that through, so
//! [`ChannelLayout::parse`] takes the expected count as a plain
//! `Option<u8>` argument instead — the same check, made explicit at the call
//! site rather than implicit in a struct nothing else here has a reason to
//! carry.

use crate::bitstream::BitReader;
use crate::error::{Error, FormatError, Result};

/// Widest channel count a single `channelLayout()` record may describe.
pub const MAX_CHANNEL_COUNT: u8 = 8;

/// CICP speaker position codes that mark a channel as LFE
/// (`impd_parse_ch_layout`'s `3 || 26` check).
const LFE_SPEAKER_POSITIONS: [u8; 2] = [3, 26];

/// A parsed `channelLayout()` record.
#[derive(Debug, Clone)]
pub struct ChannelLayout {
    pub base_channel_count: u8,
    pub layout_signaling_present: bool,
    /// Meaningful only when `layout_signaling_present`; `0` means the
    /// per-channel speaker positions below were transmitted explicitly rather
    /// than naming a standard layout.
    pub defined_layout: u8,
    /// One entry per channel, only populated when `layout_signaling_present
    /// && defined_layout == 0`; empty otherwise (a named standard layout
    /// implies its own speaker positions without transmitting them).
    pub speaker_position: Vec<u8>,
    /// One entry per channel, aligned with `speaker_position`; `true` where
    /// that channel is LFE. Empty under the same condition as
    /// `speaker_position` -- a named layout's LFE channels are implied by the
    /// layout itself, not decoded here.
    pub lfe_channel_map: Vec<bool>,
}

impl ChannelLayout {
    /// Parse one record. `expected_channel_count`, when given, must match
    /// this record's `base_channel_count` -- the reference's cross-stream
    /// consistency check between successive `channelLayout()` records that
    /// are meant to describe the same programme.
    pub fn parse(reader: &mut BitReader, expected_channel_count: Option<u8>) -> Result<Self> {
        let base_channel_count = reader.read_u8(7)?;
        if base_channel_count == 0 || base_channel_count > MAX_CHANNEL_COUNT {
            return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
                "channelLayout(): base_channel_count {base_channel_count} out of range 1..={MAX_CHANNEL_COUNT}"
            ))));
        }
        if let Some(expected) = expected_channel_count
            && expected != base_channel_count
        {
            return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
                "channelLayout(): base_channel_count {base_channel_count} disagrees with the {expected} \
                 already established for this programme"
            ))));
        }

        let layout_signaling_present = reader.read_bit()?;
        let mut defined_layout = 0u8;
        let mut speaker_position = Vec::new();
        let mut lfe_channel_map = Vec::new();

        if layout_signaling_present {
            defined_layout = reader.read_u8(8)?;
            if defined_layout == 0 {
                speaker_position.reserve(base_channel_count as usize);
                lfe_channel_map.reserve(base_channel_count as usize);
                for _ in 0..base_channel_count {
                    let position = reader.read_u8(7)?;
                    lfe_channel_map.push(LFE_SPEAKER_POSITIONS.contains(&position));
                    speaker_position.push(position);
                }
            }
        }

        Ok(Self {
            base_channel_count,
            layout_signaling_present,
            defined_layout,
            speaker_position,
            lfe_channel_map,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;

    fn encode(
        base_channel_count: u8,
        layout_signaling_present: bool,
        defined_layout: u8,
        positions: &[u8],
    ) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bits(base_channel_count as u64, 7);
        w.write_bit(layout_signaling_present);
        if layout_signaling_present {
            w.write_bits(defined_layout as u64, 8);
            if defined_layout == 0 {
                for &p in positions {
                    w.write_bits(p as u64, 7);
                }
            }
        }
        w.finalize().to_vec()
    }

    /// The common case for a named standard layout: no speaker positions are
    /// transmitted, and none should be decoded (they are implied by
    /// `defined_layout`, which this module deliberately does not interpret).
    #[test]
    fn a_named_layout_carries_no_explicit_speaker_positions() {
        let bytes = encode(6, true, 6, &[]); // e.g. "5.1" named by CICP layout index 6
        let mut r = BitReader::new(&bytes);
        let layout = ChannelLayout::parse(&mut r, None).unwrap();
        assert_eq!(layout.base_channel_count, 6);
        assert_eq!(layout.defined_layout, 6);
        assert!(layout.speaker_position.is_empty());
        assert!(layout.lfe_channel_map.is_empty());
    }

    /// An explicit layout (defined_layout == 0) must decode every channel's
    /// position, and flag exactly the two LFE position codes.
    #[test]
    fn an_explicit_layout_flags_both_lfe_position_codes() {
        let positions = [0u8, 1, 3, 4, 26, 5];
        let bytes = encode(6, true, 0, &positions);
        let mut r = BitReader::new(&bytes);
        let layout = ChannelLayout::parse(&mut r, None).unwrap();
        assert_eq!(layout.speaker_position, positions);
        assert_eq!(layout.lfe_channel_map, vec![false, false, true, false, true, false]);
    }

    /// No layout signaling at all is valid too -- just the base count.
    #[test]
    fn no_layout_signaling_leaves_positions_empty() {
        let bytes = encode(2, false, 0, &[]);
        let mut r = BitReader::new(&bytes);
        let layout = ChannelLayout::parse(&mut r, None).unwrap();
        assert_eq!(layout.base_channel_count, 2);
        assert!(!layout.layout_signaling_present);
        assert!(layout.speaker_position.is_empty());
    }

    #[test]
    fn zero_or_out_of_range_channel_count_is_rejected() {
        let bytes = encode(0, false, 0, &[]);
        let mut r = BitReader::new(&bytes);
        assert!(ChannelLayout::parse(&mut r, None).is_err());

        let bytes = encode(MAX_CHANNEL_COUNT + 1, false, 0, &[]);
        let mut r = BitReader::new(&bytes);
        assert!(ChannelLayout::parse(&mut r, None).is_err());
    }

    #[test]
    fn a_mismatched_expected_channel_count_is_rejected() {
        let bytes = encode(4, false, 0, &[]);
        let mut r = BitReader::new(&bytes);
        assert!(ChannelLayout::parse(&mut r, Some(6)).is_err());

        let bytes = encode(4, false, 0, &[]);
        let mut r = BitReader::new(&bytes);
        assert!(ChannelLayout::parse(&mut r, Some(4)).is_ok());
    }
}
