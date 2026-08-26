//! Audio Data Interchange Format (ADIF) Header Parser
//!
//! Parses ADIF file headers (ISO/IEC 13818-7 / 14496-3) containing
//! copyright identification, bitstream type, and Program Config Elements (PCE).

use crate::bitstream::{BitReader, BitWriter};
use crate::error::{FormatError, Result};
use crate::syntax::asc::AudioSpecificConfig;

/// ADIF Header Structure.
#[derive(Debug, Clone, PartialEq)]
pub struct AdifHeader {
    pub copyright_id_present: bool,
    pub copyright_id: [u8; 9],
    pub original_copy: bool,
    pub home: bool,
    pub bitstream_type: bool, // 0 = constant rate, 1 = variable rate
    pub bitrate: u32,
    pub num_program_config_elements: u8,
    pub buffer_fullness: u32,
    pub configs: Vec<AudioSpecificConfig>,
}

impl AdifHeader {
    /// ADIF syncword: 'ADIF' in ASCII (0x41444946).
    pub const SYNCWORD: u32 = 0x41444946;

    /// Parse ADIF header from bitstream reader.
    pub fn parse(reader: &mut BitReader) -> Result<Self> {
        let sync = reader.read_u32(32)?;
        if sync != Self::SYNCWORD {
            return Err(FormatError::InvalidAdif(format!("Invalid syncword: {:#010x}", sync)).into());
        }

        let copyright_id_present = reader.read_bit()?;
        let mut copyright_id = [0u8; 9];
        if copyright_id_present {
            for b in &mut copyright_id {
                *b = reader.read_u8(8)?;
            }
        }

        let original_copy = reader.read_bit()?;
        let home = reader.read_bit()?;
        let bitstream_type = reader.read_bit()?;
        let bitrate = reader.read_u32(23)?;
        let num_program_config_elements = reader.read_u8(4)? + 1;

        let buffer_fullness = if !bitstream_type {
            reader.read_u32(20)?
        } else {
            0
        };

        let mut configs = Vec::with_capacity(num_program_config_elements as usize);
        for _ in 0..num_program_config_elements {
            let asc = AudioSpecificConfig::parse(reader)?;
            configs.push(asc);
        }

        Ok(Self {
            copyright_id_present,
            copyright_id,
            original_copy,
            home,
            bitstream_type,
            bitrate,
            num_program_config_elements,
            buffer_fullness,
            configs,
        })
    }

    /// Serialize ADIF header into bitstream writer.
    pub fn write(&self, writer: &mut BitWriter) {
        writer.write_u32(Self::SYNCWORD, 32);
        writer.write_bit(self.copyright_id_present);
        if self.copyright_id_present {
            for &b in &self.copyright_id {
                writer.write_u8(b, 8);
            }
        }
        writer.write_bit(self.original_copy);
        writer.write_bit(self.home);
        writer.write_bit(self.bitstream_type);
        writer.write_u32(self.bitrate, 23);
        writer.write_u8(self.num_program_config_elements.saturating_sub(1) & 0x0F, 4);

        if !self.bitstream_type {
            writer.write_u32(self.buffer_fullness, 20);
        }

        for cfg in &self.configs {
            cfg.write(writer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AudioObjectType, ChannelConfiguration, SamplingRate};

    #[test]
    fn test_adif_header_roundtrip() {
        let config = AudioSpecificConfig {
            audio_object_type: AudioObjectType::AacLc,
            sampling_rate: SamplingRate::Hz44100,
            channel_config: ChannelConfiguration::Stereo,
            frame_length: crate::types::FrameLength::Samples1024,
            depends_on_core_coder: false,
            core_coder_delay: 0,
            extension_audio_object_type: None,
            extension_sampling_rate: None,
            sbr_present: false,
            ps_present: false,
        };

        let header = AdifHeader {
            copyright_id_present: false,
            copyright_id: [0u8; 9],
            original_copy: true,
            home: false,
            bitstream_type: true,
            bitrate: 128000,
            num_program_config_elements: 1,
            buffer_fullness: 0,
            configs: vec![config],
        };

        let mut writer = BitWriter::with_capacity(64);
        header.write(&mut writer);
        let bytes = writer.finalize();

        let mut reader = BitReader::new(bytes);
        let parsed = AdifHeader::parse(&mut reader).unwrap();

        assert_eq!(header.bitrate, parsed.bitrate);
        assert_eq!(header.num_program_config_elements, parsed.num_program_config_elements);
    }
}
