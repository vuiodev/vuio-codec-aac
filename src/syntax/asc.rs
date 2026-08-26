//! MPEG AudioSpecificConfig (ASC) and ProgramConfigElement (PCE) Parser
//!
//! Defined in ISO/IEC 14496-3 subpart 1.

use crate::bitstream::{BitReader, BitWriter};
use crate::error::{FormatError, Result};
use crate::types::{AudioObjectType, ChannelConfiguration, FrameLength, SamplingRate};

/// Parsed MPEG-4 AudioSpecificConfig (ASC).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSpecificConfig {
    pub audio_object_type: AudioObjectType,
    pub sampling_rate: SamplingRate,
    pub channel_config: ChannelConfiguration,
    pub frame_length: FrameLength,
    pub depends_on_core_coder: bool,
    pub core_coder_delay: u16,
    pub extension_audio_object_type: Option<AudioObjectType>,
    pub extension_sampling_rate: Option<SamplingRate>,
    pub sbr_present: bool,
    pub ps_present: bool,
}

impl AudioSpecificConfig {
    /// Parse an `AudioSpecificConfig` bitstream.
    pub fn parse(reader: &mut BitReader) -> Result<Self> {
        let mut aot_id = reader.read_u8(5)?;
        if aot_id == 31 {
            aot_id = 32 + reader.read_u8(6)?;
        }
        let aot = AudioObjectType::from_u8(aot_id).ok_or_else(|| {
            FormatError::InvalidAsc(format!("Unsupported Audio Object Type: {}", aot_id))
        })?;

        let sf_index = reader.read_u8(4)?;
        let sampling_rate = if sf_index == 15 {
            let custom_hz = reader.read_u32(24)?;
            SamplingRate::from_hz(custom_hz)
        } else {
            SamplingRate::from_index(sf_index).ok_or_else(|| {
                FormatError::InvalidAsc(format!("Invalid sample rate index: {}", sf_index))
            })?
        };

        let channel_config_idx = reader.read_u8(4)?;
        let channel_config = ChannelConfiguration::from_u8(channel_config_idx).ok_or_else(|| {
            FormatError::InvalidAsc(format!("Invalid channel config: {}", channel_config_idx))
        })?;

        let mut sbr_present = false;
        let mut ps_present = false;
        let mut ext_aot = None;
        let mut ext_sampling_rate = None;

        // Check for explicit SBR/PS signaling
        if aot == AudioObjectType::Sbr || aot == AudioObjectType::Ps {
            sbr_present = true;
            if aot == AudioObjectType::Ps {
                ps_present = true;
            }
            let ext_sf_idx = reader.read_u8(4)?;
            ext_sampling_rate = if ext_sf_idx == 15 {
                Some(SamplingRate::from_hz(reader.read_u32(24)?))
            } else {
                SamplingRate::from_index(ext_sf_idx)
            };
            let mut base_aot_id = reader.read_u8(5)?;
            if base_aot_id == 31 {
                base_aot_id = 32 + reader.read_u8(6)?;
            }
            ext_aot = AudioObjectType::from_u8(base_aot_id);
        }

        // Parse GA Specific Config
        let frame_length_flag = reader.read_bit()?;
        let frame_length = if frame_length_flag {
            FrameLength::Samples960
        } else {
            FrameLength::Samples1024
        };

        let depends_on_core_coder = reader.read_bit()?;
        let core_coder_delay = if depends_on_core_coder {
            reader.read_u16(14)?
        } else {
            0
        };

        let _extension_flag = reader.read_bit()?;

        Ok(Self {
            audio_object_type: aot,
            sampling_rate,
            channel_config,
            frame_length,
            depends_on_core_coder,
            core_coder_delay,
            extension_audio_object_type: ext_aot,
            extension_sampling_rate: ext_sampling_rate,
            sbr_present,
            ps_present,
        })
    }

    /// Serialize this `AudioSpecificConfig` to a `BitWriter`.
    pub fn write(&self, writer: &mut BitWriter) {
        let aot_id = self.audio_object_type.as_u8();
        if aot_id < 31 {
            writer.write_u8(aot_id, 5);
        } else {
            writer.write_u8(31, 5);
            writer.write_u8(aot_id - 32, 6);
        }

        if let Some(idx) = self.sampling_rate.to_index() {
            writer.write_u8(idx, 4);
        } else {
            writer.write_u8(15, 4);
            writer.write_u32(self.sampling_rate.hz(), 24);
        }

        writer.write_u8(self.channel_config as u8, 4);

        // GA Specific Config
        let frame_len_flag = matches!(self.frame_length, FrameLength::Samples960 | FrameLength::Samples480);
        writer.write_bit(frame_len_flag);
        writer.write_bit(self.depends_on_core_coder);
        if self.depends_on_core_coder {
            writer.write_u16(self.core_coder_delay, 14);
        }
        writer.write_bit(false); // extension_flag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asc_roundtrip() {
        let asc = AudioSpecificConfig {
            audio_object_type: AudioObjectType::AacLc,
            sampling_rate: SamplingRate::Hz48000,
            channel_config: ChannelConfiguration::Stereo,
            frame_length: FrameLength::Samples1024,
            depends_on_core_coder: false,
            core_coder_delay: 0,
            extension_audio_object_type: None,
            extension_sampling_rate: None,
            sbr_present: false,
            ps_present: false,
        };

        let mut writer = BitWriter::new();
        asc.write(&mut writer);
        let bytes = writer.finalize();

        let mut reader = BitReader::new(bytes);
        let parsed = AudioSpecificConfig::parse(&mut reader).unwrap();
        assert_eq!(asc, parsed);
    }
}
