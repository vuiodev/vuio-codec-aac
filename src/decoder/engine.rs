//! AAC decoder: raw data block parsing and the per-frame decode pipeline.
//!
//! A frame is a sequence of syntactic elements (single channels, channel pairs,
//! fill data, ...) terminated by `END`. Each coded channel goes through the same
//! chain, in the order the standard prescribes:
//!
//! 1. entropy decode and inverse quantize,
//! 2. noise substitution for bands coded as noise,
//! 3. mid/side then intensity stereo across a channel pair,
//! 4. deinterleave grouped short-window coefficients into per-window order,
//! 5. temporal noise shaping,
//! 6. IMDCT, windowing and overlap-add.
//!
//! Steps 2 to 4 are skipped where a frame does not use those tools.

use crate::bitstream::BitReader;
use crate::buffer::AudioBuffer;
use crate::decoder::aac::ics::{ChannelData, IcsInfo, NOISE_HCB, decode_ics, deinterleave};
use crate::decoder::aac::dequant::inverse_quantize_channel;
use crate::decoder::aac::pns::{NoiseMode, NoiseRng, apply_pns};
use crate::decoder::aac::stereo::{MsMask, apply_intensity_stereo, apply_ms_stereo};
use crate::decoder::aac::tns::apply_tns;
use crate::dsp::filterbank::Filterbank;
use crate::error::{DecodeError, Result};
use crate::syntax::adts::AdtsHeader;
use crate::syntax::asc::AudioSpecificConfig;
use crate::types::{
    AudioObjectType, ChannelConfiguration, FrameLength, SamplingRate, WindowShape,
};

/// Largest number of channels a single raw data block may produce.
pub const MAX_CHANNELS: usize = 8;

/// MPEG-4 audio syntactic element identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    /// Single channel element.
    Sce = 0,
    /// Channel pair element.
    Cpe = 1,
    /// Coupling channel element.
    Cce = 2,
    /// Low frequency effects element.
    Lfe = 3,
    /// Data stream element.
    Dse = 4,
    /// Program config element.
    Pce = 5,
    /// Fill element.
    Fil = 6,
    /// End of raw data block.
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

/// Per-channel state that has to survive across frames.
struct ChannelState {
    data: ChannelData,
    /// Tail of the previous frame's IMDCT, waiting to be overlap-added.
    overlap: Vec<f32>,
    /// Window shape the previous frame ended with; sets this frame's rising edge.
    prev_shape: WindowShape,
    /// Time-domain output for this frame.
    pcm: Vec<f32>,
}

impl ChannelState {
    fn new(frame_len: usize) -> Self {
        Self {
            data: ChannelData::new(frame_len),
            overlap: vec![0.0; frame_len],
            prev_shape: WindowShape::Sine,
            pcm: vec![0.0; frame_len],
        }
    }
}

/// MPEG-4 AAC decoder.
pub struct Decoder {
    config: AudioSpecificConfig,
    channels: Vec<ChannelState>,
    filterbank: Filterbank,
    output_pcm: AudioBuffer<i16>,
    frame_count: u64,
    /// Scratch for the grouped-to-per-window rearrangement.
    deinterleave_scratch: Vec<f32>,
    /// How noise substitution is seeded.
    noise_mode: NoiseMode,
    /// Generator used in [`NoiseMode::Sequential`].
    noise_rng: NoiseRng,
    /// Channels the last decoded frame actually produced.
    active_channels: usize,
}

impl Decoder {
    /// Create a decoder for the given configuration.
    pub fn new(config: AudioSpecificConfig) -> Self {
        let frame_len = config.frame_length.samples();
        let channels = (0..MAX_CHANNELS).map(|_| ChannelState::new(frame_len)).collect();
        let declared = config.channel_config.channels().max(1);

        Self {
            config,
            channels,
            filterbank: Filterbank::new(frame_len),
            output_pcm: AudioBuffer::new(declared, frame_len),
            frame_count: 0,
            deinterleave_scratch: vec![0.0; frame_len],
            noise_mode: NoiseMode::default(),
            noise_rng: NoiseRng::default(),
            active_channels: declared,
        }
    }

    /// Create a default stereo AAC-LC decoder at 44.1 kHz.
    pub fn new_default() -> Self {
        Self::new(AudioSpecificConfig {
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
        })
    }

    /// Channels the decoder is currently producing.
    pub fn channels(&self) -> usize {
        self.active_channels.max(1)
    }

    /// Output sampling rate in Hz.
    pub fn sample_rate_hz(&self) -> u32 {
        self.config.sampling_rate.hz()
    }

    /// Samples per channel per frame.
    pub fn frame_length(&self) -> usize {
        self.config.frame_length.samples()
    }

    /// Frames decoded so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Select how noise substitution is seeded; see [`NoiseMode`].
    pub fn set_noise_mode(&mut self, mode: NoiseMode) {
        self.noise_mode = mode;
    }

    /// Set the index the next decoded frame will carry.
    ///
    /// Frame indices seed noise substitution, so a decoder resuming mid-stream must
    /// be told where it is for its output to match a decode from the start.
    pub fn set_frame_index(&mut self, index: u64) {
        self.frame_count = index;
    }

    /// Reset all inter-frame state, as after a seek.
    pub fn reset(&mut self) {
        for ch in self.channels.iter_mut() {
            ch.overlap.fill(0.0);
            ch.prev_shape = WindowShape::Sine;
        }
        self.noise_rng = NoiseRng::default();
        self.frame_count = 0;
    }

    /// Generator to use for `channel` of the frame being decoded.
    ///
    /// In sequential mode this hands out the shared generator; in per-frame mode a
    /// fresh one seeded from the frame position.
    #[inline]
    fn take_noise_rng(&mut self, channel: usize) -> NoiseRng {
        match self.noise_mode {
            NoiseMode::Sequential => self.noise_rng,
            NoiseMode::PerFrame => NoiseRng::for_frame(self.frame_count, channel),
        }
    }

    /// Return a generator after use, so sequential mode keeps its position.
    #[inline]
    fn put_noise_rng(&mut self, rng: NoiseRng) {
        if self.noise_mode == NoiseMode::Sequential {
            self.noise_rng = rng;
        }
    }

    /// Decode one frame, which may carry an ADTS header or be a bare raw data block.
    pub fn decode_frame(&mut self, frame_data: &[u8]) -> Result<&AudioBuffer<i16>> {
        let mut reader = BitReader::new(frame_data);

        // An ADTS header retunes the decoder if the stream's parameters changed.
        if reader.bits_remaining() >= 56
            && let Ok(sync) = reader.peek_bits(12)
            && sync == 0x0FFF
        {
            let adts = AdtsHeader::parse(&mut reader)?;
            self.reconfigure(adts.sampling_rate, adts.channel_config, adts.audio_object_type);
        }

        let produced = self.decode_raw_data_block(&mut reader)?;
        self.active_channels = produced.max(1);

        if self.output_pcm.channels() != self.active_channels {
            self.output_pcm.resize(self.active_channels, self.frame_length());
        }

        // Convert to interleaved 16-bit PCM with saturation.
        for ch in 0..self.active_channels {
            let src = &self.channels[ch].pcm;
            let dst = self.output_pcm.channel_mut(ch);
            for (out, &v) in dst.iter_mut().zip(src.iter()) {
                *out = clamp_to_i16(v);
            }
        }

        self.frame_count += 1;
        Ok(&self.output_pcm)
    }

    /// Adopt new stream parameters, rebuilding size-dependent state if needed.
    fn reconfigure(
        &mut self,
        rate: SamplingRate,
        channel_config: ChannelConfiguration,
        aot: AudioObjectType,
    ) {
        self.config.sampling_rate = rate;
        self.config.channel_config = channel_config;
        self.config.audio_object_type = aot;
    }

    /// Parse one `raw_data_block()` and run the decode chain, returning the channel
    /// count the block produced.
    fn decode_raw_data_block(&mut self, reader: &mut BitReader) -> Result<usize> {
        let rate = self.config.sampling_rate;
        let frame_length = self.config.frame_length;
        let aot = self.config.audio_object_type;
        let mut next_channel = 0usize;

        while reader.bits_remaining() >= 3 {
            let Some(element) = ElementType::from_u8(reader.read_u8(3)?) else {
                break;
            };

            match element {
                ElementType::Sce | ElementType::Lfe => {
                    if next_channel >= MAX_CHANNELS {
                        break;
                    }
                    let _tag = reader.read_u8(4)?;
                    let mut rng = self.take_noise_rng(next_channel);
                    let ch = &mut self.channels[next_channel];
                    decode_ics(reader, &mut ch.data, rate, frame_length, aot, None)?;
                    inverse_quantize_channel(&mut ch.data);
                    apply_pns(&mut ch.data, &mut rng);
                    self.put_noise_rng(rng);
                    next_channel += 1;
                }

                ElementType::Cpe => {
                    if next_channel + 1 >= MAX_CHANNELS {
                        break;
                    }
                    let _tag = reader.read_u8(4)?;
                    let common_window = reader.read_bit()?;

                    let shared: Option<IcsInfo> = if common_window {
                        Some(IcsInfo::parse(reader, rate, frame_length, aot, true)?)
                    } else {
                        None
                    };

                    let mut mask = MsMask::default();
                    if common_window {
                        let ics = shared.as_ref().unwrap();
                        mask.kind = reader.read_u8(2)?;
                        if mask.kind == 1 {
                            for g in 0..ics.num_window_groups {
                                for sfb in 0..ics.max_sfb {
                                    mask.used[g][sfb] = reader.read_bit()?;
                                }
                            }
                        }
                    }

                    // Split so both channels of the pair can be borrowed at once.
                    let mut rng_l = self.take_noise_rng(next_channel);
                    let mut rng_r = self.take_noise_rng(next_channel + 1);
                    let sequential = self.noise_mode == NoiseMode::Sequential;
                    let (head, tail) = self.channels.split_at_mut(next_channel + 1);
                    let left = &mut head[next_channel];
                    let right = &mut tail[0];

                    decode_ics(reader, &mut left.data, rate, frame_length, aot, shared.as_ref())?;
                    decode_ics(reader, &mut right.data, rate, frame_length, aot, shared.as_ref())?;

                    inverse_quantize_channel(&mut left.data);
                    inverse_quantize_channel(&mut right.data);

                    // Noise substitution runs before the stereo tools so that M/S
                    // and intensity see complete spectra.
                    apply_pns(&mut left.data, &mut rng_l);
                    // In sequential mode both channels share one generator, so the
                    // right channel continues where the left stopped.
                    if sequential {
                        rng_r = rng_l;
                    }
                    apply_pns(&mut right.data, &mut rng_r);

                    apply_ms_stereo(&mut left.data, &mut right.data, &mask);
                    apply_intensity_stereo(&left.data, &mut right.data, &mask);

                    // The channel borrows end here, so the generator can go back.
                    self.put_noise_rng(rng_r);
                    next_channel += 2;
                }

                ElementType::Cce => {
                    // Coupling is parsed only far enough to stay bit-aligned; its
                    // gains are not applied.
                    skip_coupling_element(reader, rate, frame_length, aot)?;
                }

                ElementType::Dse => skip_data_stream_element(reader)?,
                ElementType::Pce => {
                    // A mid-stream program config would change the channel map;
                    // consume it and keep the existing configuration.
                    crate::syntax::pce::ProgramConfigElement::parse(reader)?;
                }
                ElementType::Fil => skip_fill_element(reader)?,
                ElementType::End => break,
            }
        }

        if next_channel == 0 {
            return Err(DecodeError::CorruptedFrame(
                "raw data block contained no channel elements".into(),
            )
            .into());
        }

        self.synthesize(next_channel);
        Ok(next_channel)
    }

    /// Run deinterleaving, TNS and the filterbank for every decoded channel.
    fn synthesize(&mut self, count: usize) {
        let rate = self.config.sampling_rate;

        for i in 0..count {
            let ch = &mut self.channels[i];

            // TNS and the IMDCT both work per window, so rearrange first.
            if ch.data.ics.window_sequence.is_eight_short() {
                deinterleave(&ch.data.ics, &ch.data.spec, &mut self.deinterleave_scratch);
                std::mem::swap(&mut ch.data.spec, &mut self.deinterleave_scratch);
            }

            apply_tns(&mut ch.data, rate);

            trace_frame(self.frame_count, i, &ch.data);

            let sequence = ch.data.ics.window_sequence;
            let shape = ch.data.ics.window_shape;
            let prev_shape = ch.prev_shape;

            self.filterbank.synthesize(
                &ch.data.spec,
                sequence,
                shape,
                prev_shape,
                &mut ch.overlap,
                &mut ch.pcm,
            );
            ch.prev_shape = shape;
        }
    }
}

/// Emit a one-line description of a decoded channel when `AAC_TRACE` is set.
///
/// Reports the tools the frame used and, separately, how much of its energy sits in
/// noise-substituted bands. That last figure matters when comparing against another
/// decoder: PNS bands carry only a transmitted *energy*, and each decoder synthesises
/// its own noise to fill them, so two conformant decoders necessarily disagree
/// sample-for-sample there and nowhere else.
fn trace_frame(frame: u64, channel: usize, ch: &ChannelData) {
    if std::env::var_os("AAC_TRACE").is_none() {
        return;
    }
    let ics = &ch.ics;
    let mut pns_energy = 0.0f64;
    let mut codebooks = [0usize; 16];
    for g in 0..ics.num_window_groups {
        for sfb in 0..ics.max_sfb {
            let cb = ch.sfb_cb[g][sfb];
            codebooks[cb as usize] += 1;
            if cb != NOISE_HCB {
                continue;
            }
            let lo = ics.group_base(g) + ics.grouped_offset(g, sfb);
            let width = (ics.swb_offset[sfb + 1] - ics.swb_offset[sfb]) as usize
                * ics.window_group_length[g];
            let hi = (lo + width).min(ch.spec.len());
            pns_energy += ch.spec[lo..hi].iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
        }
    }
    let total: f64 = ch.spec.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let used: Vec<String> = codebooks
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(cb, c)| format!("{cb}:{c}"))
        .collect();

    eprintln!(
        "frame {frame} ch {channel} seq {:?} shape {:?} max_sfb {} groups {} tns {} pulse {} \
pns_frac {:.4} cb[{}]",
        ics.window_sequence,
        ics.window_shape,
        ics.max_sfb,
        ics.num_window_groups,
        ch.tns.present,
        ch.pulse.is_some(),
        if total > 0.0 { pns_energy / total } else { 0.0 },
        used.join(" "),
    );
}

/// Convert a decoded sample to 16-bit PCM with saturation.
///
/// AAC spectral values are quantized against a full-scale of 32768, so the
/// filterbank already yields samples in the 16-bit range; no further scaling is
/// applied, only rounding and clipping.
#[inline(always)]
fn clamp_to_i16(v: f32) -> i16 {
    if v >= 32767.0 {
        i16::MAX
    } else if v <= -32768.0 {
        i16::MIN
    } else {
        v.round_ties_even() as i16
    }
}

/// Consume a `data_stream_element()`.
fn skip_data_stream_element(reader: &mut BitReader) -> Result<()> {
    let _tag = reader.read_u8(4)?;
    let align = reader.read_bit()?;
    let mut count = reader.read_u8(8)? as usize;
    if count == 255 {
        count += reader.read_u8(8)? as usize;
    }
    if align {
        reader.byte_align();
    }
    reader.skip_bits(count * 8)?;
    Ok(())
}

/// Consume a `fill_element()`.
///
/// Fill elements carry SBR payloads and DRC metadata, neither of which this decoder
/// applies yet; the payload still has to be stepped over exactly.
fn skip_fill_element(reader: &mut BitReader) -> Result<()> {
    let mut count = reader.read_u8(4)? as usize;
    if count == 15 {
        count += reader.read_u8(8)? as usize - 1;
    }
    reader.skip_bits(count * 8)?;
    Ok(())
}

/// Consume a `coupling_channel_element()`.
///
/// The gains and the embedded channel stream are parsed and discarded, which keeps
/// the bit position correct for whatever follows.
fn skip_coupling_element(
    reader: &mut BitReader,
    rate: SamplingRate,
    frame_length: FrameLength,
    aot: AudioObjectType,
) -> Result<()> {
    let _tag = reader.read_u8(4)?;
    let ind_sw_cce = reader.read_bit()?;
    let num_coupled = reader.read_u8(3)? as usize;

    let mut num_gain = 0usize;
    for _ in 0..=num_coupled {
        num_gain += 1;
        let cc_type = reader.read_bit()?; // channel pair flag
        let _id = reader.read_u8(4)?;
        if cc_type {
            let _ch_select = reader.read_u8(2)?;
            num_gain += 1;
        }
    }

    let _cc_domain = reader.read_bit()?;
    let _gain_element_sign = reader.read_bit()?;
    let _gain_element_scale = reader.read_u8(2)?;

    let mut data = ChannelData::new(frame_length.samples());
    decode_ics(reader, &mut data, rate, frame_length, aot, None)?;

    // One gain list per coupled target, minus the implicit first when the coupling
    // is independently switched.
    let start = if ind_sw_cce { 1 } else { 0 };
    for c in start..num_gain {
        let mut cge = true;
        if c != 0 {
            cge = reader.read_bit()?;
        }
        if cge {
            skip_scalefactor_code(reader)?;
        } else {
            for g in 0..data.ics.num_window_groups {
                for sfb in 0..data.ics.max_sfb {
                    if data.sfb_cb[g][sfb] != 0 {
                        skip_scalefactor_code(reader)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Consume one Huffman-coded scalefactor delta.
fn skip_scalefactor_code(reader: &mut BitReader) -> Result<()> {
    crate::decoder::aac::huffman::decode_scalefactor_delta(reader)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_reports_its_configuration() {
        let d = Decoder::new_default();
        assert_eq!(d.channels(), 2);
        assert_eq!(d.sample_rate_hz(), 44100);
        assert_eq!(d.frame_length(), 1024);
        assert_eq!(d.frame_count(), 0);
    }

    /// Element identifiers must round-trip through the enum.
    #[test]
    fn element_ids_round_trip() {
        for id in 0..8u8 {
            let e = ElementType::from_u8(id).expect("all three-bit ids are defined");
            assert_eq!(e as u8, id);
        }
    }

    /// PCM conversion must saturate rather than wrap.
    #[test]
    fn pcm_conversion_saturates() {
        assert_eq!(clamp_to_i16(40000.0), i16::MAX);
        assert_eq!(clamp_to_i16(-40000.0), i16::MIN);
        assert_eq!(clamp_to_i16(0.0), 0);
        assert_eq!(clamp_to_i16(16384.4), 16384);
        assert_eq!(clamp_to_i16(-16384.4), -16384);
    }

    /// A frame with no channel elements is an error, not silence.
    #[test]
    fn empty_block_is_rejected() {
        let mut d = Decoder::new_default();
        // A single END element and nothing else.
        let data = [0xE0u8, 0, 0, 0, 0, 0, 0, 0];
        assert!(d.decode_frame(&data).is_err());
    }

    /// Decoding must not panic on arbitrary bytes.
    #[test]
    fn arbitrary_input_does_not_panic() {
        let mut d = Decoder::new_default();
        for seed in 0..256u32 {
            let data: Vec<u8> = (0..64)
                .map(|i| ((seed.wrapping_mul(2654435761).wrapping_add(i)) >> 5) as u8)
                .collect();
            let _ = d.decode_frame(&data);
        }
    }
}
