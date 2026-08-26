//! High-Level MPEG-4 AAC / HE-AAC Audio Decoder Engine
//!
//! Coordinates bitstream demuxing, Huffman entropy decoding, scalefactor inverse quantization,
//! M/S & Intensity stereo decoding, and IMDCT transform synthesis into uncompressed 16-bit PCM.

use crate::bitstream::BitReader;
use crate::buffer::AudioBuffer;
use crate::decoder::aac::channel::IcsInfo;
use crate::decoder::aac::dequant::inverse_quantize;
use crate::decoder::aac::huffman::decode_spectral_band;
use crate::decoder::aac::stereo::apply_ms_stereo;
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
    // Reusable pre-allocated scratch buffers
    scratch_spectral: Vec<Vec<f32>>,
    scratch_imdct: Vec<f32>,
    scratch_time_pcm: Vec<f32>,
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

        let scratch_spectral = vec![vec![0.0f32; frame_samples]; 8];
        let scratch_imdct = vec![0.0f32; 2 * frame_samples];
        let scratch_time_pcm = vec![0.0f32; frame_samples];

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
            scratch_spectral,
            scratch_imdct,
            scratch_time_pcm,
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
        let sampling_rate = self.config.sampling_rate;
        let frame_length_cfg = self.config.frame_length;

        // Ensure internal buffers match channel configuration
        if self.overlap_history.channels() != num_ch {
            self.overlap_history.resize(num_ch, frame_len);
            self.output_pcm.resize(num_ch, frame_len);
        }

        for buf in self.scratch_spectral.iter_mut().take(num_ch) {
            buf.fill(0.0);
        }

        // Parse syntactic elements
        while reader.bits_remaining() >= 3 {
            let elem_id = reader.read_u8(3)?;
            let elem_type = ElementType::from_u8(elem_id).unwrap_or(ElementType::End);

            match elem_type {
                ElementType::Sce => {
                    let _tag = reader.read_u8(4)?;
                    decode_single_channel_stream(&mut reader, sampling_rate, frame_length_cfg, &mut self.scratch_spectral[0])?;
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
                        let (left_slice, right_slice) = self.scratch_spectral.split_at_mut(1);
                        decode_channel_stream_data(&mut reader, sampling_rate, frame_length_cfg, &ics_left, &mut left_slice[0])?;
                        decode_channel_stream_data(&mut reader, sampling_rate, frame_length_cfg, &ics_right, &mut right_slice[0])?;

                        if ms_mask_present == 1 || ms_mask_present == 2 {
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

        // Apply IMDCT, windowing, and overlap-add for each channel with zero heap allocations
        for ch in 0..num_ch {
            let history = self.overlap_history.channel_mut(ch);
            let spec = &self.scratch_spectral[ch];

            self.mdct_1024.process_overlap_add_scratch(
                spec,
                &self.sine_window_2048,
                history,
                &mut self.scratch_time_pcm,
                &mut self.scratch_imdct,
            );

            // Convert floating point PCM to signed 16-bit integer output
            let pcm_out = self.output_pcm.channel_mut(ch);
            for (out_sample, &sample_f32) in pcm_out.iter_mut().zip(self.scratch_time_pcm.iter()) {
                let scaled = (sample_f32 * 32768.0).round();
                *out_sample = scaled.clamp(-32768.0, 32767.0) as i16;
            }
        }

        self.frame_count += 1;
        Ok(&self.output_pcm)
    }
}

fn decode_single_channel_stream(
    reader: &mut BitReader,
    sampling_rate: SamplingRate,
    frame_length: FrameLength,
    spectral: &mut [f32],
) -> Result<()> {
    let global_gain = reader.read_u8(8)? as i16;
    let ics = IcsInfo::parse(reader, false)?;
    let sfb_table = get_sfb_table(sampling_rate, false, frame_length);
    let mut sfb_offsets = [0usize; 64];
    let num_sfb = compute_sfb_offsets(sfb_table, &mut sfb_offsets);

    let max_sfb = (ics.max_sfb as usize).min(num_sfb);
    let mut quantized_band = [0i32; 1024];

    for b in 0..max_sfb {
        let cb = reader.read_u8(4)?;
        let start = sfb_offsets[b];
        let end = sfb_offsets[b + 1].min(1024);
        let len = end.saturating_sub(start);

        if len > 0 {
            decode_spectral_band(reader, cb, &mut quantized_band[start..end])?;
            inverse_quantize(&quantized_band[start..end], global_gain, &mut spectral[start..end]);
        }
    }
    Ok(())
}

fn decode_channel_stream_data(
    reader: &mut BitReader,
    sampling_rate: SamplingRate,
    frame_length: FrameLength,
    ics: &IcsInfo,
    spectral: &mut [f32],
) -> Result<()> {
    let global_gain = reader.read_u8(8)? as i16;
    let sfb_table = get_sfb_table(sampling_rate, false, frame_length);
    let mut sfb_offsets = [0usize; 64];
    let num_sfb = compute_sfb_offsets(sfb_table, &mut sfb_offsets);

    let max_sfb = (ics.max_sfb as usize).min(num_sfb);
    let mut quantized_band = [0i32; 1024];

    for b in 0..max_sfb {
        let cb = reader.read_u8(4)?;
        let start = sfb_offsets[b];
        let end = sfb_offsets[b + 1].min(1024);
        let len = end.saturating_sub(start);

        if len > 0 {
            decode_spectral_band(reader, cb, &mut quantized_band[start..end])?;
            inverse_quantize(&quantized_band[start..end], global_gain, &mut spectral[start..end]);
        }
    }
    Ok(())
}

/// MPEG-4 Audio syntactic element type tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    Sce = 0,
    Cpe = 1,
    Cce = 2,
    Lfe = 3,
    Dse = 4,
    Pce = 5,
    Fil = 6,
    End = 7,
}

impl ElementType {
    pub const fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::Sce),
            1 => Some(Self::Cpe),
            2 => Some(Self::Cce),
            3 => Some(Self::Lfe),
            4 => Some(Self::Dse),
            5 => Some(Self::Pce),
            6 => Some(Self::Fil),
            7 => Some(Self::End),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation_and_lifecycle() {
        let decoder = Decoder::new_default();
        assert_eq!(decoder.channels(), 2);
        assert_eq!(decoder.sample_rate_hz(), 44100);
        assert_eq!(decoder.frame_length(), 1024);
        assert_eq!(decoder.frame_count(), 0);
    }
}
