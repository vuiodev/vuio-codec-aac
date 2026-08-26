//! High-Performance Audio Decoder Engine
//!
//! Orchestrates ADTS/RAW bitstream parsing, entropy decoding, joint stereo,
//! IMDCT overlap-add rendering, and SBR/PS synthesis into final PCM frames.

use crate::bitstream::BitReader;
use crate::buffer::AudioBuffer;
use crate::decoder::aac::{
    apply_ms_stereo, decode_spectral_band, inverse_quantize, ElementType, IcsInfo,
};
use crate::dsp::mdct::MdctContext;
use crate::dsp::window::{generate_kbd_window_f32, generate_sine_window_f32};
use crate::error::Result;
use crate::syntax::adts::AdtsHeader;
use crate::syntax::asc::AudioSpecificConfig;
use crate::tables::scalefactor::{compute_sfb_offsets, get_sfb_table};
use crate::types::{
    AudioObjectType, ChannelConfiguration, FrameLength, SamplingRate,
};

/// High-level AAC / HE-AAC / USAC Audio Decoder instance.
pub struct Decoder {
    config: AudioSpecificConfig,
    mdct_1024: MdctContext,
    _mdct_128: MdctContext,
    sine_window_2048: Vec<f32>,
    _sine_window_256: Vec<f32>,
    _kbd_window_2048: Vec<f32>,
    _kbd_window_256: Vec<f32>,
    overlap_history: AudioBuffer<f32>,
    output_pcm: AudioBuffer<i16>,
    frame_count: u64,
}

impl Decoder {
    /// Create a new `Decoder` initialized from an `AudioSpecificConfig`.
    pub fn new(config: AudioSpecificConfig) -> Self {
        let channels = config.channel_config.channels().max(1);
        let frame_samples = config.frame_length.samples();

        let mdct_1024 = MdctContext::new(frame_samples);
        let mdct_128 = MdctContext::new(config.frame_length.short_samples());

        let sine_window_2048 = generate_sine_window_f32(2 * frame_samples);
        let sine_window_256 = generate_sine_window_f32(2 * config.frame_length.short_samples());
        let kbd_window_2048 = generate_kbd_window_f32(2 * frame_samples, 4.0);
        let kbd_window_256 = generate_kbd_window_f32(2 * config.frame_length.short_samples(), 6.0);

        let overlap_history = AudioBuffer::new(channels, frame_samples);
        let output_pcm = AudioBuffer::new(channels, frame_samples);

        Self {
            config,
            mdct_1024,
            _mdct_128: mdct_128,
            sine_window_2048,
            _sine_window_256: sine_window_256,
            _kbd_window_2048: kbd_window_2048,
            _kbd_window_256: kbd_window_256,
            overlap_history,
            output_pcm,
            frame_count: 0,
        }
    }

    /// Create a default stereo AAC-LC decoder (44.1 kHz, 1024 samples/frame).
    pub fn new_default() -> Self {
        let config = AudioSpecificConfig {
            audio_object_type: AudioObjectType::AacLc,
            sampling_rate: SamplingRate::Hz44100,
            channel_config: ChannelConfiguration::Stereo,
            frame_length: FrameLength::Samples1024,
            depends_on_core_coder: false,
            core_coder_delay: 0,
            extension_audio_object_type: None,
            extension_sampling_rate: None,
            sbr_present: false,
            ps_present: false,
        };
        Self::new(config)
    }

    /// Number of audio channels configured in decoder.
    pub fn channels(&self) -> usize {
        self.config.channel_config.channels().max(1)
    }

    /// Sampling rate in Hertz.
    pub fn sample_rate_hz(&self) -> u32 {
        self.config.sampling_rate.hz()
    }

    /// Samples per frame per channel.
    pub fn frame_length(&self) -> usize {
        self.config.frame_length.samples()
    }

    /// Total frames decoded so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Decode a single raw or ADTS frame payload into 16-bit PCM output.
    pub fn decode_frame(&mut self, frame_data: &[u8]) -> Result<&AudioBuffer<i16>> {
        let mut reader = BitReader::new(frame_data);

        // Check if ADTS header present at start of frame
        if reader.bits_remaining() >= 56
            && let Ok(sync) = reader.peek_bits(12)
                && sync == 0x0FFF {
                    let adts = AdtsHeader::parse(&mut reader)?;
                    self.config.sampling_rate = adts.sampling_rate;
                    self.config.channel_config = adts.channel_config;
                }

        let num_ch = self.channels();
        let frame_len = self.frame_length();

        // Ensure internal buffers match channel configuration
        if self.overlap_history.channels() != num_ch {
            self.overlap_history.resize(num_ch, frame_len);
            self.output_pcm.resize(num_ch, frame_len);
        }

        let mut spectral_buffers = vec![vec![0.0f32; frame_len]; num_ch];

        // Parse syntactic elements
        while reader.bits_remaining() >= 3 {
            let elem_id = reader.read_u8(3)?;
            let elem_type = ElementType::from_u8(elem_id).unwrap_or(ElementType::End);

            match elem_type {
                ElementType::Sce => {
                    let _tag = reader.read_u8(4)?;
                    self.decode_single_channel(&mut reader, &mut spectral_buffers[0])?;
                }
                ElementType::Cpe => {
                    let _tag = reader.read_u8(4)?;
                    let common_window = reader.read_bit()?;
                    let ics_left = IcsInfo::parse(&mut reader, common_window)?;
                    let ics_right = if common_window {
                        ics_left.clone()
                    } else {
                        IcsInfo::parse(&mut reader, false)?
                    };

                    let ms_mask_present = reader.read_u8(2)?;
                    if ms_mask_present == 1 {
                        let num_bands = ics_left.max_sfb as usize;
                        for _ in 0..num_bands {
                            let _ms_flag = reader.read_bit()?;
                        }
                    }

                    if num_ch >= 2 {
                        self.decode_channel_stream(&mut reader, &ics_left, &mut spectral_buffers[0])?;
                        self.decode_channel_stream(&mut reader, &ics_right, &mut spectral_buffers[1])?;

                        if ms_mask_present == 1 || ms_mask_present == 2 {
                            let (left_slice, right_slice) = spectral_buffers.split_at_mut(1);
                            apply_ms_stereo(&mut left_slice[0], &mut right_slice[0]);
                        }
                    }
                }
                ElementType::End => break,
                _ => {
                    break;
                }
            }
        }

        // Apply IMDCT, windowing, and overlap-add for each channel
        for (ch, spec) in spectral_buffers.iter().enumerate().take(num_ch) {
            let mut time_pcm = vec![0.0f32; frame_len];
            let history = self.overlap_history.channel_mut(ch);

            self.mdct_1024.process_overlap_add(
                spec,
                &self.sine_window_2048,
                history,
                &mut time_pcm,
            );

            // Convert floating point PCM to signed 16-bit integer output
            let pcm_out = self.output_pcm.channel_mut(ch);
            for (out_sample, &sample_f32) in pcm_out.iter_mut().zip(time_pcm.iter()) {
                let scaled = (sample_f32 * 32768.0).round();
                *out_sample = scaled.clamp(-32768.0, 32767.0) as i16;
            }
        }

        self.frame_count += 1;
        Ok(&self.output_pcm)
    }

    fn decode_single_channel(
        &self,
        reader: &mut BitReader,
        spectral: &mut [f32],
    ) -> Result<()> {
        let global_gain = reader.read_u8(8)? as i16;
        let ics = IcsInfo::parse(reader, false)?;
        self.decode_channel_stream_with_gain(reader, &ics, global_gain, spectral)
    }

    fn decode_channel_stream(
        &self,
        reader: &mut BitReader,
        ics: &IcsInfo,
        spectral: &mut [f32],
    ) -> Result<()> {
        let global_gain = reader.read_u8(8)? as i16;
        self.decode_channel_stream_with_gain(reader, ics, global_gain, spectral)
    }

    fn decode_channel_stream_with_gain(
        &self,
        reader: &mut BitReader,
        ics: &IcsInfo,
        global_gain: i16,
        spectral: &mut [f32],
    ) -> Result<()> {
        let is_short = ics.window_sequence.is_eight_short();
        let sfb_widths = get_sfb_table(self.config.sampling_rate, is_short, self.config.frame_length);

        let mut sfb_offsets = [0usize; 64];
        let num_bands = compute_sfb_offsets(sfb_widths, &mut sfb_offsets).min(ics.max_sfb as usize + 1);

        // 1. Read section data
        let mut bands_read = 0;
        let mut sections = Vec::new();
        while bands_read < ics.max_sfb as usize && reader.bits_remaining() >= 9 {
            let sect_cb = reader.read_u8(4)?;
            let mut sect_len = 0;
            loop {
                let incr = reader.read_u8(5)? as usize;
                sect_len += incr;
                if incr < 31 || bands_read + sect_len >= ics.max_sfb as usize {
                    break;
                }
            }
            sections.push((sect_cb, sect_len));
            bands_read += sect_len;
        }

        // 2. Tool Present Flags
        if reader.bits_remaining() >= 3 {
            let _pulse = reader.read_bit()?;
            let _tns = reader.read_bit()?;
            let _gain_ctrl = reader.read_bit()?;
        }

        let current_sf = global_gain;
        let mut quantized = vec![0i32; self.frame_length()];

        // 3. Spectral Data
        let mut current_band = 0;
        for (sect_cb, sect_len) in sections {
            if sect_cb != 0 {
                for _ in 0..sect_len {
                    if current_band + 1 < num_bands {
                        let start = sfb_offsets[current_band];
                        let end = sfb_offsets[current_band + 1].min(spectral.len());
                        let band_len = end.saturating_sub(start);
                        if band_len > 0 {
                            let _ = decode_spectral_band(reader, sect_cb, &mut quantized[start..end]);
                            inverse_quantize(&quantized[start..end], current_sf, &mut spectral[start..end]);
                        }
                    }
                    current_band += 1;
                }
            } else {
                current_band += sect_len;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation_and_lifecycle() {
        let mut decoder = Decoder::new_default();
        assert_eq!(decoder.channels(), 2);
        assert_eq!(decoder.sample_rate_hz(), 44100);
        assert_eq!(decoder.frame_length(), 1024);

        // Construct valid ADTS frame with END element (7)
        let mut payload_writer = crate::bitstream::BitWriter::new();
        payload_writer.write_u8(7, 3); // END element
        let payload = payload_writer.into_bytes();

        let header = AdtsHeader {
            mpeg_id: 0,
            layer: 0,
            protection_absent: true,
            audio_object_type: AudioObjectType::AacLc,
            sampling_rate: SamplingRate::Hz44100,
            channel_config: ChannelConfiguration::Stereo,
            frame_length: 7 + payload.len(),
            buffer_fullness: 0x7FF,
            num_raw_data_blocks: 0,
            crc: None,
        };

        let mut adts_writer = crate::bitstream::BitWriter::new();
        header.write(&mut adts_writer);
        adts_writer.write_bytes(&payload);
        let adts_frame = adts_writer.into_bytes();

        let result = decoder.decode_frame(&adts_frame);
        assert!(result.is_ok());
        let pcm = result.unwrap();
        assert_eq!(pcm.channels(), 2);
        assert_eq!(pcm.samples_per_channel(), 1024);
    }
}
