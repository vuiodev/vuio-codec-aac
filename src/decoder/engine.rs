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
use crate::decoder::drc::{DrcDecoder, DrcInfo, DrcSettings};
use crate::decoder::sbr::{SBR_CORE_FRAME, SbrDecoder, SbrElement};
use crate::dsp::filterbank::Filterbank;
use crate::error::{DecodeError, Result};
use crate::syntax::adts::AdtsHeader;
use crate::syntax::asc::AudioSpecificConfig;
use crate::types::{
    AudioObjectType, ChannelConfiguration, FrameLength, SamplingRate, WindowShape,
};

/// Largest number of channels a single raw data block may produce.
pub const MAX_CHANNELS: usize = 8;

/// Extension payload identifiers that may appear in a fill element.
mod extension {
    /// Dynamic range control metadata.
    pub const DYNAMIC_RANGE: u8 = 11;
    /// Spectral band replication payload.
    pub const SBR_DATA: u8 = 13;
    /// The same, preceded by a CRC over it.
    pub const SBR_DATA_CRC: u8 = 14;
}

/// Where an SBR payload found in a fill element applies.
///
/// SBR data rides in a fill element that follows the channel element it describes,
/// so parsing has to remember what came just before.
#[derive(Debug, Clone, Copy)]
struct SbrTarget {
    /// Index of the channel element among the frame's channel elements.
    element: usize,
    /// Whether that element carries one channel or two.
    kind: SbrElement,
}

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
    /// Band-replicated output, at twice the core rate.
    sbr_pcm: Vec<f32>,
}

impl ChannelState {
    fn new(frame_len: usize) -> Self {
        Self {
            data: ChannelData::new(frame_len),
            overlap: vec![0.0; frame_len],
            prev_shape: WindowShape::Sine,
            pcm: vec![0.0; frame_len],
            sbr_pcm: Vec::new(),
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
    /// One band replicator per channel element, created when the first payload for
    /// that element arrives.
    sbr: Vec<SbrDecoder>,
    /// Set once any element has carried an SBR payload; from then on the decoder
    /// runs the replication chain every frame, so its delay stays constant.
    sbr_active: bool,
    /// Which channels each channel element of the last frame produced.
    elements: Vec<(usize, usize)>,
    /// Holds a channel's core frame while the replicator writes its output.
    sbr_scratch: Vec<f32>,
    /// Dynamic range control, which does nothing until a listener asks for it.
    drc: DrcDecoder,
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
            sbr: Vec::new(),
            sbr_active: false,
            elements: Vec::new(),
            sbr_scratch: Vec::new(),
            drc: DrcDecoder::default(),
        }
    }

    /// Say what to do with any dynamic range metadata the stream carries.
    ///
    /// The default honours none of it, which decodes a stream carrying DRC exactly
    /// as it would one that did not.
    pub fn set_drc_settings(&mut self, settings: DrcSettings) {
        self.drc.set_settings(settings);
    }

    /// What the decoder is currently doing with dynamic range metadata.
    #[inline]
    pub fn drc_settings(&self) -> DrcSettings {
        self.drc.settings()
    }

    /// Whether the stream has been found to carry dynamic range metadata.
    #[inline]
    pub fn drc_present(&self) -> bool {
        self.drc.is_present()
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
    ///
    /// Band replication doubles it, so a stream whose headers say 22.05 kHz decodes
    /// to 44.1 kHz once its first SBR payload has been seen.
    pub fn sample_rate_hz(&self) -> u32 {
        let core = self.config.sampling_rate.hz();
        if self.sbr_active { core * 2 } else { core }
    }

    /// Sampling rate of the AAC core, before any band replication.
    pub fn core_sample_rate_hz(&self) -> u32 {
        self.config.sampling_rate.hz()
    }

    /// Samples per channel per frame, at the output rate.
    pub fn frame_length(&self) -> usize {
        let core = self.config.frame_length.samples();
        if self.sbr_active { core * 2 } else { core }
    }

    /// Whether the stream has been found to carry band replication.
    pub fn sbr_active(&self) -> bool {
        self.sbr_active
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
        for sbr in &mut self.sbr {
            sbr.reset();
        }
        self.drc.reset();
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

        if self.drc.is_active() && !self.sbr_active {
            for ch in 0..self.active_channels.min(MAX_CHANNELS) {
                self.drc.apply_to_spectrum(ch, &mut self.channels[ch].pcm);
            }
        }

        if self.sbr_active {
            self.apply_band_replication()?;
            // Parametric stereo may have widened the frame past what the core
            // decoded, so the channel count is only final once it has run.
            self.active_channels = self.replicated_channels().clamp(1, MAX_CHANNELS);
            if self.drc.is_active() {
                for ch in 0..self.active_channels {
                    self.drc.apply_to_samples(ch, &mut self.channels[ch].sbr_pcm);
                }
            }
        }

        let frame_len = self.frame_length();
        if self.output_pcm.channels() != self.active_channels
            || self.output_pcm.samples_per_channel() != frame_len
        {
            self.output_pcm.resize(self.active_channels, frame_len);
        }

        // Convert to interleaved 16-bit PCM with saturation.
        for ch in 0..self.active_channels {
            let src: &[f32] =
                if self.sbr_active { &self.channels[ch].sbr_pcm } else { &self.channels[ch].pcm };
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
        let mut sbr_target: Option<SbrTarget> = None;
        self.elements.clear();

        while reader.bits_remaining() >= 3 {
            let Some(element) = ElementType::from_u8(reader.read_u8(3)?) else {
                break;
            };

            match element {
                ElementType::Sce | ElementType::Lfe => {
                    if next_channel >= MAX_CHANNELS {
                        break;
                    }
                    if element == ElementType::Sce {
                        sbr_target =
                            Some(SbrTarget { element: self.elements.len(), kind: SbrElement::Single });
                    }
                    self.elements.push((next_channel, 1));
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
                    sbr_target =
                        Some(SbrTarget { element: self.elements.len(), kind: SbrElement::Pair });
                    self.elements.push((next_channel, 2));
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
                ElementType::Fil => self.parse_fill_element(reader, sbr_target)?,
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

    /// Consume a `fill_element()`, handing any SBR payload to the replicator that
    /// owns the channel element it follows.
    ///
    /// Whatever the payload turns out to be, the reader is left exactly at the end
    /// of the declared byte count: a payload this decoder does not understand, or
    /// one that fails to parse, must not desynchronise the rest of the frame.
    fn parse_fill_element(
        &mut self,
        reader: &mut BitReader,
        target: Option<SbrTarget>,
    ) -> Result<()> {
        let mut count = reader.read_u8(4)? as usize;
        if count == 15 {
            count += reader.read_u8(8)? as usize;
            count -= 1;
        }
        if count == 0 {
            return Ok(());
        }
        let payload_bits = count * 8;
        if reader.bits_remaining() < payload_bits {
            return Err(DecodeError::CorruptedFrame("fill element runs past the frame".into()).into());
        }
        let start = reader.bit_position();

        let kind = reader.read_u8(4)?;
        match kind {
            extension::SBR_DATA | extension::SBR_DATA_CRC => {
                if let Some(target) = target {
                    let with_crc = kind == extension::SBR_DATA_CRC;
                    self.decode_sbr_payload(reader, target, with_crc);
                }
            }
            extension::DYNAMIC_RANGE => {
                // Metadata that fails to parse is dropped: the audio is unaffected,
                // and the previous frame's settings hold until the next good payload.
                if let Ok(info) = DrcInfo::parse(reader) {
                    self.drc.accept(info);
                }
            }
            _ => {}
        }

        let consumed = reader.bit_position() - start;
        if consumed < payload_bits {
            reader.skip_bits(payload_bits - consumed)?;
        }
        Ok(())
    }

    /// Hand one SBR payload to the right replicator, creating it if this is the
    /// first payload that element has carried.
    ///
    /// A payload that fails to parse is dropped rather than propagated: the core
    /// signal is still perfectly good, and the alternative is discarding a frame
    /// over metadata the ear would barely notice.
    fn decode_sbr_payload(&mut self, reader: &mut BitReader, target: SbrTarget, with_crc: bool) {
        if self.config.frame_length.samples() != SBR_CORE_FRAME {
            return;
        }
        let core_rate = self.config.sampling_rate.hz();
        while self.sbr.len() <= target.element {
            self.sbr.push(SbrDecoder::new(target.kind.channels(), core_rate, false));
        }
        let sbr = &mut self.sbr[target.element];
        sbr.set_core_rate(core_rate);
        match sbr.decode_extension(reader, target.kind, with_crc) {
            Ok(()) => self.sbr_active = true,
            Err(e) => {
                if std::env::var_os("AAC_TRACE_SBR").is_some() {
                    eprintln!("sbr payload rejected: {e}");
                }
            }
        }
    }

    /// Output channels the elements of the last frame add up to.
    ///
    /// Parametric stereo turns a single-channel element into two, so this can
    /// exceed the number of channels the core decoded.
    fn replicated_channels(&self) -> usize {
        self.elements
            .iter()
            .enumerate()
            .map(|(index, &(_, count))| {
                self.sbr.get(index).map_or(count, |sbr| sbr.output_channels().max(count))
            })
            .sum()
    }

    /// Run the replication chain for every channel of every element that has one.
    ///
    /// Elements with no replicator still have their sample rate doubled, or the
    /// frame would mix channels running at two different rates.
    fn apply_band_replication(&mut self) -> Result<()> {
        let core_len = self.config.frame_length.samples();
        let out_len = core_len * 2;

        let produced = self.replicated_channels().max(self.active_channels);
        for ch in self.channels.iter_mut().take(produced.min(MAX_CHANNELS)) {
            if ch.sbr_pcm.len() != out_len {
                ch.sbr_pcm = vec![0.0; out_len];
            }
        }

        // Every element gets a replicator once any element has one, so that all
        // channels take the same filterbank path and share the same delay. A
        // replicator with no payload simply resamples.
        let core_rate = self.config.sampling_rate.hz();
        for &(_, count) in &self.elements {
            if self.sbr.len() < self.elements.len() {
                self.sbr.push(SbrDecoder::new(count, core_rate, false));
            }
        }

        // The core frame and the replicated frame both live in `channels`, so the
        // core is moved into a scratch buffer for the call and moved back after.
        let elements = std::mem::take(&mut self.elements);
        let mut result = Ok(());
        let mut out_channel = 0usize;
        for (index, &(first, count)) in elements.iter().enumerate() {
            // A parametric stereo element reads one coded channel and writes two.
            if self.sbr.get(index).is_some_and(SbrDecoder::parametric_stereo)
                && out_channel + 1 < MAX_CHANNELS
            {
                std::mem::swap(&mut self.sbr_scratch, &mut self.channels[first].pcm);
                let (head, tail) = self.channels.split_at_mut(out_channel + 1);
                let mut left = std::mem::take(&mut head[out_channel].sbr_pcm);
                let mut right = std::mem::take(&mut tail[0].sbr_pcm);
                if let Err(e) = self.sbr[index].process_parametric(
                    &self.sbr_scratch,
                    &mut left,
                    &mut right,
                ) && result.is_ok()
                {
                    result = Err(e);
                }
                self.channels[out_channel].sbr_pcm = left;
                self.channels[out_channel + 1].sbr_pcm = right;
                std::mem::swap(&mut self.sbr_scratch, &mut self.channels[first].pcm);
                out_channel += 2;
                continue;
            }

            for offset in 0..count {
                let channel = first + offset;
                if channel >= self.active_channels {
                    continue;
                }
                out_channel = out_channel.max(channel + 1);
                std::mem::swap(&mut self.sbr_scratch, &mut self.channels[channel].pcm);
                let mut produced = std::mem::take(&mut self.channels[channel].sbr_pcm);

                match self.sbr.get_mut(index) {
                    Some(sbr) => {
                        if let Err(e) = sbr.process_channel(offset, &self.sbr_scratch, &mut produced)
                            && result.is_ok()
                        {
                            result = Err(e);
                        }
                    }
                    None => {
                        // No replicator at all: hold each sample for two output
                        // samples. Only reachable for a frame whose element count
                        // grew mid-stream.
                        for (i, &v) in self.sbr_scratch.iter().enumerate() {
                            produced[2 * i] = v;
                            produced[2 * i + 1] = v;
                        }
                    }
                }

                self.channels[channel].sbr_pcm = produced;
                std::mem::swap(&mut self.sbr_scratch, &mut self.channels[channel].pcm);
            }
        }
        self.elements = elements;
        result
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
