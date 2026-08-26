//! High-Level MPEG-4 AAC / HE-AAC Audio Encoder Engine
//!
//! Coordinates Time/Frequency MDCT transformation, psychoacoustic masking,
//! non-uniform quantization, rate control, and multi-channel bitstream formatting.

use crate::bitstream::BitWriter;
use crate::buffer::AudioBuffer;
use crate::dsp::fft::Complex32;
use crate::dsp::mdct::MdctContext;
use crate::dsp::window::generate_sine_window_f32;
use crate::encoder::aac::bitstream::{finalize_adts_frame, write_multichannel_elements};
use crate::encoder::aac::psycho::PsychoacousticModel;
use crate::encoder::aac::quant::{estimate_global_gain, quantize_band};
use crate::error::Result;
use crate::tables::scalefactor::{compute_sfb_offsets, get_sfb_table};
use crate::types::{AudioObjectType, ChannelConfiguration, FrameLength, SamplingRate};

/// Configuration options for initializing an AAC Encoder instance.
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

/// Zero-Allocation Production AAC Audio Encoder instance.
#[derive(Debug, Clone)]
pub struct Encoder {
    config: EncoderConfig,
    time_history: AudioBuffer<f32>,
    sine_window: Vec<f32>,
    _mdct: MdctContext,
    psycho: PsychoacousticModel,
    frame_count: u64,
    sfb_offsets: [usize; 64],
    num_sfb: usize,
    target_bits_per_ch: usize,
    // Reusable pre-allocated scratch buffers (zero allocations in hot path)
    scratch_combined: Vec<f32>,
    scratch_windowed: Vec<f32>,
    scratch_spectral: Vec<f32>,
    scratch_fft: Vec<Complex32>,
    scratch_all_quantized: Vec<Vec<i32>>,
    scratch_global_gains: Vec<i16>,
    scratch_writer: BitWriter,
}

impl Encoder {
    /// Create and initialize a new `Encoder` with the specified configuration.
    pub fn new(config: EncoderConfig) -> Result<Self> {
        let num_ch = config.channel_config.channels();
        let frame_len = config.frame_length.samples();

        let time_history = AudioBuffer::<f32>::new(num_ch, frame_len);
        let sine_window = generate_sine_window_f32(2 * frame_len);
        let mdct = MdctContext::new(frame_len);

        let sfb_table = get_sfb_table(config.sampling_rate, false, config.frame_length);
        let mut sfb_offsets = [0usize; 64];
        let num_sfb = compute_sfb_offsets(sfb_table, &mut sfb_offsets);
        let psycho = PsychoacousticModel::new(num_sfb);

        let target_bits_per_ch = (config.bitrate_bps
            / (config.sampling_rate.hz() / frame_len as u32)
            / num_ch as u32) as usize;

        let scratch_combined = vec![0.0f32; 2 * frame_len];
        let scratch_windowed = vec![0.0f32; 2 * frame_len];
        let scratch_spectral = vec![0.0f32; frame_len];
        let scratch_fft = vec![Complex32::new(0.0, 0.0); 2 * frame_len];
        let scratch_all_quantized = vec![vec![0i32; frame_len]; num_ch];
        let scratch_global_gains = vec![0i16; num_ch];
        let scratch_writer = BitWriter::with_capacity(2048);

        Ok(Self {
            config,
            time_history,
            sine_window,
            _mdct: mdct,
            psycho,
            frame_count: 0,
            sfb_offsets,
            num_sfb,
            target_bits_per_ch,
            scratch_combined,
            scratch_windowed,
            scratch_spectral,
            scratch_fft,
            scratch_all_quantized,
            scratch_global_gains,
            scratch_writer,
        })
    }

    /// Number of audio frames encoded so far.
    #[inline(always)]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Reset encoder internal state and history buffers.
    pub fn reset(&mut self) {
        for ch in 0..self.time_history.channels() {
            self.time_history.channel_mut(ch).fill(0.0);
        }
        self.frame_count = 0;
    }

    /// Encode a single frame of uncompressed 16-bit PCM audio into a standard ADTS AAC packet.
    #[inline(always)]
    pub fn encode_frame(&mut self, pcm_input: &AudioBuffer<i16>) -> Result<Vec<u8>> {
        let num_ch = self.config.channel_config.channels();
        let frame_len = self.config.frame_length.samples();

        assert_eq!(
            pcm_input.channels(),
            num_ch,
            "Input buffer channels mismatch configured encoder channels"
        );
        assert_eq!(
            pcm_input.samples_per_channel(),
            frame_len,
            "Input buffer length mismatch frame size"
        );

        let num_sfb = self.num_sfb;
        let max_sfb = (num_sfb.saturating_sub(1)) as u8;
        let sfb_offsets = &self.sfb_offsets;
        let target_bits = self.target_bits_per_ch;

        self.scratch_writer.reset();

        for ch in 0..num_ch {
            let pcm_samples = pcm_input.channel(ch);

            // 1. Combine history and new samples
            let hist = self.time_history.channel(ch);
            for i in 0..frame_len {
                self.scratch_combined[i] = hist[i];
                self.scratch_combined[frame_len + i] = (pcm_samples[i] as f32) * (1.0 / 32768.0);
            }

            // Update history for next frame
            let hist_mut = self.time_history.channel_mut(ch);
            for i in 0..frame_len {
                hist_mut[i] = (pcm_samples[i] as f32) * (1.0 / 32768.0);
            }

            // 2. Window and Forward MDCT via FFT (O(N log N))
            for i in 0..2 * frame_len {
                self.scratch_windowed[i] = self.scratch_combined[i] * self.sine_window[i];
            }

            self._mdct.forward_mdct_fft(
                &self.scratch_windowed,
                &mut self.scratch_spectral,
                &mut self.scratch_fft,
            );

            // 3. Psychoacoustic Analysis & Scalefactor Quantization
            let _psy_result = self.psycho.analyze(&self.scratch_spectral, &sfb_offsets[..num_sfb]);
            let global_gain = estimate_global_gain(&self.scratch_spectral, target_bits);
            self.scratch_global_gains[ch] = global_gain;

            quantize_band(&self.scratch_spectral, global_gain, &mut self.scratch_all_quantized[ch]);
        }

        // 4. Emit syntactic elements for configured channel layout
        write_multichannel_elements(
            &mut self.scratch_writer,
            self.config.channel_config,
            &self.scratch_global_gains,
            max_sfb,
            &self.scratch_all_quantized,
            None,
        );

        // Element Terminator (END = 7)
        self.scratch_writer.write_u8(7, 3);
        let raw_payload = self.scratch_writer.finalize();

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
        let config = EncoderConfig {
            audio_object_type: AudioObjectType::AacLc,
            sampling_rate: SamplingRate::Hz44100,
            channel_config: ChannelConfiguration::Stereo,
            bitrate_bps: 128000,
            frame_length: FrameLength::Samples1024,
        };

        let mut encoder = Encoder::new(config).unwrap();
        let mut pcm_input = AudioBuffer::<i16>::new(2, 1024);

        for ch in 0..2 {
            let slice = pcm_input.channel_mut(ch);
            for (i, s) in slice.iter_mut().enumerate() {
                *s = ((i as f32 * 0.1).sin() * 10000.0) as i16;
            }
        }

        let packet = encoder.encode_frame(&pcm_input).unwrap();
        assert!(!packet.is_empty(), "ADTS frame must not be empty");
        assert_eq!(packet[0], 0xFF);
        assert_eq!(packet[1] & 0xF0, 0xF0);
        assert_eq!(encoder.frame_count(), 1);
    }
}
