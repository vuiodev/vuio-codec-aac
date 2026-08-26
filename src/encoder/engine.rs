//! High-Performance Audio Encoder Engine
//!
//! Orchestrates transient detection, forward MDCT, psychoacoustic analysis,
//! rate-distortion quantization, and ADTS bitstream emission.

use crate::bitstream::BitWriter;
use crate::buffer::AudioBuffer;
use crate::dsp::mdct::MdctContext;
use crate::dsp::window::generate_sine_window_f32;
use crate::encoder::aac::{
    estimate_global_gain, finalize_adts_frame, quantize_band, write_multichannel_elements,
    BlockSwitching, PsychoacousticModel,
};
use crate::error::{EncodeError, Result};
use crate::tables::scalefactor::{compute_sfb_offsets, get_sfb_table};
use crate::types::{AudioObjectType, ChannelConfiguration, FrameLength, SamplingRate};

/// Configuration parameters for initializing an `Encoder`.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub audio_object_type: AudioObjectType,
    pub sampling_rate: SamplingRate,
    pub channel_config: ChannelConfiguration,
    pub bitrate_bps: u32,
    pub frame_length: FrameLength,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            audio_object_type: AudioObjectType::AacLc,
            sampling_rate: SamplingRate::Hz44100,
            channel_config: ChannelConfiguration::Stereo,
            bitrate_bps: 128000,
            frame_length: FrameLength::Samples1024,
        }
    }
}

/// High-level AAC Audio Encoder instance.
pub struct Encoder {
    config: EncoderConfig,
    _mdct: MdctContext,
    sine_window: Vec<f32>,
    psycho: PsychoacousticModel,
    _block_switch: Vec<BlockSwitching>,
    time_history: AudioBuffer<f32>,
    frame_count: u64,
}

impl Encoder {
    /// Create a new `Encoder` with the specified configuration.
    pub fn new(config: EncoderConfig) -> Result<Self> {
        let channels = config.channel_config.channels().max(1);
        let frame_samples = config.frame_length.samples();

        let mdct = MdctContext::new(frame_samples);
        let sine_window = generate_sine_window_f32(2 * frame_samples);
        let psycho = PsychoacousticModel::new(64);
        let block_switch = vec![BlockSwitching::new(); channels];
        let time_history = AudioBuffer::new(channels, frame_samples);

        Ok(Self {
            config,
            _mdct: mdct,
            sine_window,
            psycho,
            _block_switch: block_switch,
            time_history,
            frame_count: 0,
        })
    }

    /// Number of audio channels configured in encoder.
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

    /// Total frames encoded so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Encode a single multi-channel 16-bit PCM audio frame into an ADTS packet.
    pub fn encode_frame(&mut self, pcm_input: &AudioBuffer<i16>) -> Result<Vec<u8>> {
        let num_ch = self.channels();
        let frame_len = self.frame_length();

        if pcm_input.channels() < num_ch || pcm_input.samples_per_channel() < frame_len {
            return Err(EncodeError::InvalidInputSize {
                provided: pcm_input.samples_per_channel(),
                required: frame_len,
            }
            .into());
        }

        let sfb_widths = get_sfb_table(self.config.sampling_rate, false, self.config.frame_length);
        let mut sfb_offsets = [0usize; 64];
        let num_sfb = compute_sfb_offsets(sfb_widths, &mut sfb_offsets);
        let max_sfb = (num_sfb.saturating_sub(1)) as u8;

        let mut raw_writer = BitWriter::with_capacity(1024);
        let mut all_quantized = vec![vec![0i32; frame_len]; num_ch];
        let mut global_gains = vec![0i16; num_ch];

        for ch in 0..num_ch {
            let pcm_samples = pcm_input.channel(ch);
            let mut combined_time = vec![0.0f32; 2 * frame_len];

            // 1. Combine history and new samples
            let hist = self.time_history.channel(ch);
            for i in 0..frame_len {
                combined_time[i] = hist[i];
                combined_time[frame_len + i] = (pcm_samples[i] as f32) / 32768.0;
            }

            // Update history for next frame
            let hist_mut = self.time_history.channel_mut(ch);
            for i in 0..frame_len {
                hist_mut[i] = (pcm_samples[i] as f32) / 32768.0;
            }

            // 2. Window and Forward MDCT
            let mut windowed = vec![0.0f32; 2 * frame_len];
            for (w, (&t, &win)) in windowed.iter_mut().zip(combined_time.iter().zip(self.sine_window.iter())) {
                *w = t * win;
            }

            let mut spectral = vec![0.0f32; frame_len];
            self._mdct.forward_mdct(&windowed, &mut spectral);

            // 3. Psychoacoustic Analysis & Scalefactor Quantization
            let _psy_result = self.psycho.analyze(&spectral, &sfb_offsets[..num_sfb]);
            let target_bits = (self.config.bitrate_bps / (self.config.sampling_rate.hz() / frame_len as u32) / num_ch as u32) as usize;
            let global_gain = estimate_global_gain(&spectral, target_bits);
            global_gains[ch] = global_gain;

            quantize_band(&spectral, global_gain, &mut all_quantized[ch]);
        }

        // 4. Emit syntactic elements for configured channel layout
        write_multichannel_elements(
            &mut raw_writer,
            self.config.channel_config,
            &global_gains,
            max_sfb,
            &all_quantized,
            None,
        );

        // Element Terminator (END = 7)
        raw_writer.write_u8(7, 3);
        let raw_payload = raw_writer.finalize();

        // 5. Wrap in ADTS framing
        let adts_frame = finalize_adts_frame(
            raw_payload,
            self.config.audio_object_type,
            self.config.sampling_rate,
            self.config.channel_config,
        );

        self.frame_count += 1;
        Ok(adts_frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation_and_frame_encoding() {
        let mut encoder = Encoder::new(EncoderConfig::default()).unwrap();
        assert_eq!(encoder.channels(), 2);
        assert_eq!(encoder.sample_rate_hz(), 44100);

        let pcm = AudioBuffer::<i16>::new(2, 1024);
        let adts_packet = encoder.encode_frame(&pcm).unwrap();

        assert!(!adts_packet.is_empty());
        assert_eq!(adts_packet[0], 0xFF);
        assert_eq!(adts_packet[1] & 0xF0, 0xF0);
    }
}
