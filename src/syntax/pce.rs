//! Program Config Element (PCE) Parser and Serializer
//!
//! Handles custom multi-channel speaker mapping, LFE counts, matrix mixdown coefficients,
//! and pseudo-surround metadata (ISO/IEC 13818-7 / 14496-3 Section 4.5.1.1).

use crate::bitstream::{BitReader, BitWriter};
use crate::error::Result;
use crate::types::{AudioObjectType, SamplingRate};

/// Program Config Element definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramConfigElement {
    pub element_instance_tag: u8,
    pub object_type: AudioObjectType,
    pub sampling_rate: SamplingRate,
    pub num_front_channel_elements: u8,
    pub num_side_channel_elements: u8,
    pub num_back_channel_elements: u8,
    pub num_lfe_channel_elements: u8,
    pub num_assoc_data_elements: u8,
    pub num_valid_cc_elements: u8,
    pub mono_mixdown_present: bool,
    pub mono_mixdown_element_number: u8,
    pub stereo_mixdown_present: bool,
    pub stereo_mixdown_element_number: u8,
    pub matrix_mixdown_idx_present: bool,
    pub matrix_mixdown_idx: u8,
    pub pseudo_surround_enable: bool,
    pub front_elements: Vec<(bool, u8)>, // (is_cpe, tag)
    pub side_elements: Vec<(bool, u8)>,
    pub back_elements: Vec<(bool, u8)>,
    pub lfe_element_tags: Vec<u8>,
}

impl ProgramConfigElement {
    /// Parse Program Config Element from bitstream reader.
    pub fn parse(reader: &mut BitReader) -> Result<Self> {
        let element_instance_tag = reader.read_u8(4)?;
        let aot_id = reader.read_u8(2)? + 1;
        let object_type = AudioObjectType::from_u8(aot_id).unwrap_or(AudioObjectType::AacLc);
        let sr_idx = reader.read_u8(4)?;
        let sampling_rate = SamplingRate::from_index(sr_idx).unwrap_or(SamplingRate::Hz44100);

        let num_front_channel_elements = reader.read_u8(4)?;
        let num_side_channel_elements = reader.read_u8(4)?;
        let num_back_channel_elements = reader.read_u8(4)?;
        let num_lfe_channel_elements = reader.read_u8(2)?;
        let num_assoc_data_elements = reader.read_u8(3)?;
        let num_valid_cc_elements = reader.read_u8(4)?;

        let mono_mixdown_present = reader.read_bit()?;
        let mono_mixdown_element_number = if mono_mixdown_present {
            reader.read_u8(4)?
        } else {
            0
        };

        let stereo_mixdown_present = reader.read_bit()?;
        let stereo_mixdown_element_number = if stereo_mixdown_present {
            reader.read_u8(4)?
        } else {
            0
        };

        let matrix_mixdown_idx_present = reader.read_bit()?;
        let (matrix_mixdown_idx, pseudo_surround_enable) = if matrix_mixdown_idx_present {
            (reader.read_u8(2)?, reader.read_bit()?)
        } else {
            (0, false)
        };

        let mut front_elements = Vec::with_capacity(num_front_channel_elements as usize);
        for _ in 0..num_front_channel_elements {
            front_elements.push((reader.read_bit()?, reader.read_u8(4)?));
        }

        let mut side_elements = Vec::with_capacity(num_side_channel_elements as usize);
        for _ in 0..num_side_channel_elements {
            side_elements.push((reader.read_bit()?, reader.read_u8(4)?));
        }

        let mut back_elements = Vec::with_capacity(num_back_channel_elements as usize);
        for _ in 0..num_back_channel_elements {
            back_elements.push((reader.read_bit()?, reader.read_u8(4)?));
        }

        let mut lfe_element_tags = Vec::with_capacity(num_lfe_channel_elements as usize);
        for _ in 0..num_lfe_channel_elements {
            lfe_element_tags.push(reader.read_u8(4)?);
        }

        Ok(Self {
            element_instance_tag,
            object_type,
            sampling_rate,
            num_front_channel_elements,
            num_side_channel_elements,
            num_back_channel_elements,
            num_lfe_channel_elements,
            num_assoc_data_elements,
            num_valid_cc_elements,
            mono_mixdown_present,
            mono_mixdown_element_number,
            stereo_mixdown_present,
            stereo_mixdown_element_number,
            matrix_mixdown_idx_present,
            matrix_mixdown_idx,
            pseudo_surround_enable,
            front_elements,
            side_elements,
            back_elements,
            lfe_element_tags,
        })
    }

    /// Serialize Program Config Element into bitstream writer.
    pub fn write(&self, writer: &mut BitWriter) {
        writer.write_u8(self.element_instance_tag & 0x0F, 4);
        let aot_val = (self.object_type.as_u8().saturating_sub(1)) & 0x03;
        writer.write_u8(aot_val, 2);
        writer.write_u8(self.sampling_rate.index() & 0x0F, 4);

        writer.write_u8(self.num_front_channel_elements & 0x0F, 4);
        writer.write_u8(self.num_side_channel_elements & 0x0F, 4);
        writer.write_u8(self.num_back_channel_elements & 0x0F, 4);
        writer.write_u8(self.num_lfe_channel_elements & 0x03, 2);
        writer.write_u8(self.num_assoc_data_elements & 0x07, 3);
        writer.write_u8(self.num_valid_cc_elements & 0x0F, 4);

        writer.write_bit(self.mono_mixdown_present);
        if self.mono_mixdown_present {
            writer.write_u8(self.mono_mixdown_element_number & 0x0F, 4);
        }

        writer.write_bit(self.stereo_mixdown_present);
        if self.stereo_mixdown_present {
            writer.write_u8(self.stereo_mixdown_element_number & 0x0F, 4);
        }

        writer.write_bit(self.matrix_mixdown_idx_present);
        if self.matrix_mixdown_idx_present {
            writer.write_u8(self.matrix_mixdown_idx & 0x03, 2);
            writer.write_bit(self.pseudo_surround_enable);
        }

        for &(is_cpe, tag) in &self.front_elements {
            writer.write_bit(is_cpe);
            writer.write_u8(tag & 0x0F, 4);
        }
        for &(is_cpe, tag) in &self.side_elements {
            writer.write_bit(is_cpe);
            writer.write_u8(tag & 0x0F, 4);
        }
        for &(is_cpe, tag) in &self.back_elements {
            writer.write_bit(is_cpe);
            writer.write_u8(tag & 0x0F, 4);
        }
        for &tag in &self.lfe_element_tags {
            writer.write_u8(tag & 0x0F, 4);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pce_roundtrip() {
        let pce = ProgramConfigElement {
            element_instance_tag: 0,
            object_type: AudioObjectType::AacLc,
            sampling_rate: SamplingRate::Hz44100,
            num_front_channel_elements: 1,
            num_side_channel_elements: 0,
            num_back_channel_elements: 0,
            num_lfe_channel_elements: 0,
            num_assoc_data_elements: 0,
            num_valid_cc_elements: 0,
            mono_mixdown_present: false,
            mono_mixdown_element_number: 0,
            stereo_mixdown_present: false,
            stereo_mixdown_element_number: 0,
            matrix_mixdown_idx_present: false,
            matrix_mixdown_idx: 0,
            pseudo_surround_enable: false,
            front_elements: vec![(true, 0)],
            side_elements: vec![],
            back_elements: vec![],
            lfe_element_tags: vec![],
        };

        let mut writer = BitWriter::with_capacity(32);
        pce.write(&mut writer);
        let bytes = writer.finalize();

        let mut reader = BitReader::new(bytes);
        let parsed = ProgramConfigElement::parse(&mut reader).unwrap();

        assert_eq!(pce.element_instance_tag, parsed.element_instance_tag);
        assert_eq!(pce.front_elements.len(), parsed.front_elements.len());
    }
}
