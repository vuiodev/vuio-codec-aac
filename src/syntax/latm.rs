//! Low Overhead Audio Stream (LOAS) / LATM Transport Demuxer
//!
//! Parses LOAS sync headers (0x2B7, 11 bits) and LATM `AudioMuxElement` payloads
//! (ISO/IEC 14496-3 Part 3 Subpart 1) for broadcast and streaming audio.

use crate::bitstream::{BitReader, BitWriter};
use crate::error::{Result, SyntaxError};
use crate::syntax::asc::AudioSpecificConfig;

/// LOAS / LATM Audio Multiplex Element.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioMuxElement {
    pub mux_config_present: bool,
    pub stream_mux_config: Option<StreamMuxConfig>,
    pub payload_bytes: Vec<u8>,
}

/// Stream Multiplex Configuration contained in LATM headers.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamMuxConfig {
    pub audio_mux_version: u8,
    pub all_streams_same_time_framing: bool,
    pub num_sub_frames: u8,
    pub num_programs: u8,
    pub num_layers: u8,
    pub asc: AudioSpecificConfig,
}

impl AudioMuxElement {
    /// LOAS Syncword: 0x2B7 (11 bits).
    pub const LOAS_SYNCWORD: u16 = 0x2B7;

    /// Parse LOAS audio packet including syncword, frame length, and AudioMuxElement.
    pub fn parse_loas(reader: &mut BitReader) -> Result<Self> {
        let sync = reader.read_u16(11)?;
        if sync != Self::LOAS_SYNCWORD {
            return Err(SyntaxError::InvalidSyncword { syncword: sync as u32 }.into());
        }

        let audio_mux_length = reader.read_u16(13)? as usize;
        let mux_config_present = reader.read_bit()?;

        let stream_mux_config = if mux_config_present {
            Some(StreamMuxConfig::parse(reader)?)
        } else {
            None
        };

        // Read payload bytes
        let mut payload_bytes = Vec::new();
        let remaining_bits = reader.bits_remaining().min(audio_mux_length * 8);
        let num_bytes = remaining_bits / 8;
        for _ in 0..num_bytes {
            payload_bytes.push(reader.read_u8(8)?);
        }

        Ok(Self {
            mux_config_present,
            stream_mux_config,
            payload_bytes,
        })
    }

    /// Write LOAS formatted packet into bitstream writer.
    pub fn write_loas(&self, writer: &mut BitWriter) {
        writer.write_u16(Self::LOAS_SYNCWORD, 11);
        let length = self.payload_bytes.len() + if self.mux_config_present { 4 } else { 1 };
        writer.write_u16(length as u16, 13);
        writer.write_bit(self.mux_config_present);

        if let Some(ref config) = self.stream_mux_config {
            config.write(writer);
        }

        for &b in &self.payload_bytes {
            writer.write_u8(b, 8);
        }
    }
}

impl StreamMuxConfig {
    /// Parse StreamMuxConfig from bitstream reader.
    pub fn parse(reader: &mut BitReader) -> Result<Self> {
        let audio_mux_version = reader.read_u8(1)?;
        let all_streams_same_time_framing = reader.read_bit()?;
        let num_sub_frames = reader.read_u8(6)? + 1;
        let num_programs = reader.read_u8(4)? + 1;
        let num_layers = reader.read_u8(3)? + 1;

        let asc = AudioSpecificConfig::parse(reader)?;

        Ok(Self {
            audio_mux_version,
            all_streams_same_time_framing,
            num_sub_frames,
            num_programs,
            num_layers,
            asc,
        })
    }

    /// Write StreamMuxConfig into bitstream writer.
    pub fn write(&self, writer: &mut BitWriter) {
        writer.write_u8(self.audio_mux_version & 1, 1);
        writer.write_bit(self.all_streams_same_time_framing);
        writer.write_u8(self.num_sub_frames.saturating_sub(1) & 0x3F, 6);
        writer.write_u8(self.num_programs.saturating_sub(1) & 0x0F, 4);
        writer.write_u8(self.num_layers.saturating_sub(1) & 0x07, 3);
        self.asc.write(writer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AudioObjectType, ChannelConfiguration, SamplingRate};

    #[test]
    fn test_loas_roundtrip() {
        let asc = AudioSpecificConfig {
            audio_object_type: AudioObjectType::AacLc,
            sampling_rate: SamplingRate::Hz48000,
            channel_config: ChannelConfiguration::Stereo,
            frame_length: crate::types::FrameLength::Samples1024,
            sbr_present: false,
            ps_present: false,
        };

        let config = StreamMuxConfig {
            audio_mux_version: 0,
            all_streams_same_time_framing: true,
            num_sub_frames: 1,
            num_programs: 1,
            num_layers: 1,
            asc,
        };

        let elem = AudioMuxElement {
            mux_config_present: true,
            stream_mux_config: Some(config),
            payload_bytes: vec![0x11, 0x22, 0x33, 0x44],
        };

        let mut writer = BitWriter::with_capacity(64);
        elem.write_loas(&mut writer);
        let bytes = writer.finalize();

        let mut reader = BitReader::new(bytes);
        let parsed = AudioMuxElement::parse_loas(&mut reader).unwrap();

        assert_eq!(parsed.mux_config_present, true);
        assert_eq!(parsed.payload_bytes.len(), 4);
    }
}
