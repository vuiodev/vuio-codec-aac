//! Audio Data Transport Stream (ADTS) Header Parsing and Serialization
//!
//! Defined in ISO/IEC 13818-7 / ISO/IEC 14496-3.

use crate::bitstream::{BitReader, BitWriter};
use crate::error::{FormatError, Result};
use crate::types::{AudioObjectType, ChannelConfiguration, SamplingRate};

pub const ADTS_SYNCWORD: u16 = 0x0FFF;
pub const ADTS_HEADER_SIZE_NO_CRC: usize = 7;
pub const ADTS_HEADER_SIZE_CRC: usize = 9;

/// Parsed ADTS Frame Header (Fixed + Variable header fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdtsHeader {
    /// MPEG ID (0 for MPEG-4, 1 for MPEG-2).
    pub mpeg_id: u8,
    /// Layer (always 0 in AAC).
    pub layer: u8,
    /// Protection absent (1 if no CRC-16, 0 if 16-bit CRC follows header).
    pub protection_absent: bool,
    /// Audio Object Type (derived from profile + 1).
    pub audio_object_type: AudioObjectType,
    /// Sampling rate.
    pub sampling_rate: SamplingRate,
    /// Channel configuration.
    pub channel_config: ChannelConfiguration,
    /// Total frame length in bytes including header and CRC.
    pub frame_length: usize,
    /// ADTS buffer fullness (0x7FF for VBR).
    pub buffer_fullness: u16,
    /// Number of raw data blocks in frame minus 1 (0 means 1 block/frame).
    pub num_raw_data_blocks: u8,
    /// Optional 16-bit CRC checksum if `protection_absent` is false.
    pub crc: Option<u16>,
}

impl AdtsHeader {
    /// Parse an ADTS header from a `BitReader`.
    pub fn parse(reader: &mut BitReader) -> Result<Self> {
        let syncword = reader.read_u16(12)?;
        if syncword != ADTS_SYNCWORD {
            return Err(FormatError::InvalidAdts(format!(
                "Invalid ADTS syncword: {:#05x} (expected 0x0FFF)",
                syncword
            ))
            .into());
        }

        let mpeg_id = reader.read_u8(1)?;
        let layer = reader.read_u8(2)?;
        let protection_absent = reader.read_bit()?;
        let profile = reader.read_u8(2)?;
        let aot = AudioObjectType::from_u8(profile + 1).unwrap_or(AudioObjectType::AacLc);

        let sf_index = reader.read_u8(4)?;
        let sampling_rate = SamplingRate::from_index(sf_index).ok_or_else(|| {
            FormatError::InvalidAdts(format!("Invalid sampling frequency index: {}", sf_index))
        })?;

        let _private_bit = reader.read_bit()?;
        let channel_config_idx = reader.read_u8(3)?;
        let channel_config = ChannelConfiguration::from_u8(channel_config_idx).ok_or_else(|| {
            FormatError::InvalidAdts(format!("Invalid channel config: {}", channel_config_idx))
        })?;

        let _original_copy = reader.read_bit()?;
        let _home = reader.read_bit()?;

        // Variable Header
        let _copyright_id_bit = reader.read_bit()?;
        let _copyright_id_start = reader.read_bit()?;
        let frame_length = reader.read_u16(13)? as usize;
        let buffer_fullness = reader.read_u16(11)?;
        let num_raw_data_blocks = reader.read_u8(2)?;

        let crc = if !protection_absent {
            Some(reader.read_u16(16)?)
        } else {
            None
        };

        Ok(Self {
            mpeg_id,
            layer,
            protection_absent,
            audio_object_type: aot,
            sampling_rate,
            channel_config,
            frame_length,
            buffer_fullness,
            num_raw_data_blocks,
            crc,
        })
    }

    /// Serialize this ADTS header to a `BitWriter`.
    pub fn write(&self, writer: &mut BitWriter) {
        writer.write_bits(ADTS_SYNCWORD as u64, 12);
        writer.write_u8(self.mpeg_id, 1);
        writer.write_u8(self.layer, 2);
        writer.write_bit(self.protection_absent);

        let profile = (self.audio_object_type.as_u8().saturating_sub(1)) & 0x03;
        writer.write_u8(profile, 2);

        let sf_index = self.sampling_rate.to_index().unwrap_or(4);
        writer.write_u8(sf_index, 4);
        writer.write_bit(false); // private bit

        writer.write_u8(self.channel_config as u8, 3);
        writer.write_bit(false); // original_copy
        writer.write_bit(false); // home

        writer.write_bit(false); // copyright_id_bit
        writer.write_bit(false); // copyright_id_start
        writer.write_u16(self.frame_length as u16, 13);
        writer.write_u16(self.buffer_fullness, 11);
        writer.write_u8(self.num_raw_data_blocks, 2);

        if let Some(crc_val) = self.crc {
            writer.write_u16(crc_val, 16);
        }
    }

    /// Size of this header in bytes (7 without CRC, 9 with CRC).
    pub const fn header_size(&self) -> usize {
        if self.protection_absent {
            ADTS_HEADER_SIZE_NO_CRC
        } else {
            ADTS_HEADER_SIZE_CRC
        }
    }

    /// Size of raw audio payload in bytes (`frame_length - header_size`).
    pub fn payload_size(&self) -> usize {
        self.frame_length.saturating_sub(self.header_size())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adts_header_roundtrip() {
        let header = AdtsHeader {
            mpeg_id: 0,
            layer: 0,
            protection_absent: true,
            audio_object_type: AudioObjectType::AacLc,
            sampling_rate: SamplingRate::Hz44100,
            channel_config: ChannelConfiguration::Stereo,
            frame_length: 350,
            buffer_fullness: 0x7FF,
            num_raw_data_blocks: 0,
            crc: None,
        };

        let mut writer = BitWriter::new();
        header.write(&mut writer);
        let bytes = writer.finalize();

        assert_eq!(bytes.len(), 7);
        let mut reader = BitReader::new(bytes);
        let parsed = AdtsHeader::parse(&mut reader).unwrap();
        assert_eq!(header, parsed);
    }
}
