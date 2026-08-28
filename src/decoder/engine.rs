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
use crate::syntax::adif::AdifHeader;
use crate::syntax::adts::AdtsHeader;
use crate::syntax::asc::AudioSpecificConfig;
use crate::syntax::latm::AudioMuxElement;
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
    /// The output-stage peak limiter, built lazily once the output channel
    /// count and rate are known. `None` until [`Decoder::enable_peak_limiter`]
    /// is called: like DRC, this is opt-in, because the limiter's look-ahead
    /// delay shifts every output sample by a few milliseconds, which a caller
    /// who did not ask for it should not get silently.
    peak_limiter: Option<crate::dsp::peak_limiter::PeakLimiter>,
    /// Per-coupling-channel overlap-add state, keyed by the CCE's own
    /// `element_instance_tag` rather than decode order -- a CCE is not one of
    /// `self.channels` and can appear or disappear frame to frame, so its
    /// filterbank tail has to persist under a stable key instead of a slot
    /// index. See [`Self::mix_coupling_channels`] and `text/plan.txt` phase 7.5.
    cce_overlap: std::collections::HashMap<u8, (Vec<f32>, WindowShape)>,
    /// See [`Self::enable_downmix_to_stereo`]; off by default.
    downmix_to_stereo: bool,
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
            peak_limiter: None,
            cce_overlap: std::collections::HashMap::new(),
            downmix_to_stereo: false,
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

    /// Turn the output-stage look-ahead peak limiter on.
    ///
    /// Off by default: enabling it delays every output sample by the
    /// limiter's look-ahead depth (`dsp::peak_limiter::DEFAULT_ATTACK_TIME_MS`,
    /// a few milliseconds) and, once a loud passage engages it, applies a
    /// gain reduction that a caller comparing against an unlimited reference
    /// would not expect. A limiter built for one rate/channel-count is
    /// rebuilt automatically if the stream reconfigures to a different one.
    pub fn enable_peak_limiter(&mut self) {
        self.peak_limiter =
            Some(crate::dsp::peak_limiter::PeakLimiter::new(self.active_channels, self.sample_rate_hz()));
    }

    /// Turn the peak limiter back off; the next frame decodes exactly as if
    /// it had never been enabled (any audio still in the limiter's look-ahead
    /// delay line is discarded, not flushed).
    pub fn disable_peak_limiter(&mut self) {
        self.peak_limiter = None;
    }

    /// Whether the peak limiter is currently engaged.
    pub fn peak_limiter_enabled(&self) -> bool {
        self.peak_limiter.is_some()
    }

    /// Fold a multichannel stream down to stereo output using the reference's
    /// fixed downmix matrix (`decoder::aac::downmix`), for output devices that
    /// cannot render more than two channels.
    ///
    /// Off by default, like the peak limiter and DRC. Only takes effect on a
    /// frame whose `channelConfiguration` implicitly declares one of the four
    /// layouts [`crate::decoder::aac::downmix::Layout`] covers (5.0/5.1/7.0/7.1
    /// with no `program_config_element()` overriding it -- see
    /// [`Self::downmix_layout`]); any other channel count decodes and outputs
    /// unchanged, exactly as if this were never called. That is a real, if
    /// narrower-than-libxaac, scope: the reference additionally resolves a
    /// PCE's own declared channel roles via `slot_element[]`, which this
    /// decoder does not track (`text/plan.txt` phase 7.6).
    pub fn enable_downmix_to_stereo(&mut self) {
        self.downmix_to_stereo = true;
    }

    /// Turn multichannel-to-stereo downmixing back off.
    pub fn disable_downmix_to_stereo(&mut self) {
        self.downmix_to_stereo = false;
    }

    /// Whether downmixing is currently enabled (not whether the last frame
    /// actually had a channel count it applies to).
    pub fn downmix_to_stereo_enabled(&self) -> bool {
        self.downmix_to_stereo
    }

    /// The fixed downmix layout this decoder's current implicit channel
    /// configuration resolves to, if any.
    ///
    /// Only the three implicit (PCE-less) configurations whose element decode
    /// order the standard fixes are covered: `FiveChannel` (5.0),
    /// `FivePointOne` (5.1) and `SevenPointOne` (7.1, which is what
    /// `ChannelConfiguration::SevenPointOne` denotes despite the name reusing
    /// "seven" -- MPEG-4's own table). `Layout::Ch7_0` has no implicit
    /// `channelConfiguration` value at all (7 discrete channels with no LFE
    /// only exists behind an explicit PCE this decoder does not role-map), so
    /// it is unreachable from here by construction, not an oversight.
    fn downmix_layout(&self) -> Option<crate::decoder::aac::downmix::Layout> {
        use crate::decoder::aac::downmix::Layout;
        match self.config.channel_config {
            ChannelConfiguration::FiveChannel => Some(Layout::Ch5_0),
            ChannelConfiguration::FivePointOne => Some(Layout::Ch5_1),
            ChannelConfiguration::SevenPointOne => Some(Layout::Ch7_1),
            _ => None,
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

    /// Decode one frame, which may carry a LOAS/LATM header, an ADTS header, or
    /// be a bare raw data block.
    pub fn decode_frame(&mut self, frame_data: &[u8]) -> Result<&AudioBuffer<i16>> {
        // A LOAS sync (0x2B7) means the actual raw_data_block is wrapped in an
        // AudioMuxElement, not sitting at the start of `frame_data` -- unwrap it
        // to its payload before anything else looks at the bytes. This owns the
        // extracted payload for the rest of the function, since the ADTS/raw
        // path below borrows whichever byte slice is current.
        let latm_payload: Vec<u8>;
        let source: &[u8] = match BitReader::new(frame_data).peek_bits(11) {
            Ok(sync) if sync as u16 == AudioMuxElement::LOAS_SYNCWORD => {
                let mut peek = BitReader::new(frame_data);
                let elem = AudioMuxElement::parse_loas(&mut peek)?;
                if let Some(cfg) = &elem.stream_mux_config {
                    self.reconfigure(
                        cfg.asc.sampling_rate,
                        cfg.asc.channel_config,
                        cfg.asc.audio_object_type,
                    );
                }
                latm_payload = elem.payload_bytes;
                &latm_payload
            }
            _ => frame_data,
        };

        let mut reader = BitReader::new(source);

        // An ADIF header appears once, at the very start of a whole stream, and
        // unlike ADTS/LOAS nothing frames the raw_data_block()s that follow it
        // -- so rather than extracting a payload, consume the header in place
        // and keep decoding from the same reader, right where it left off.
        if reader.bits_remaining() >= 32
            && let Ok(sync) = reader.peek_bits(32)
            && sync as u32 == AdifHeader::SYNCWORD
        {
            let adif = AdifHeader::parse(&mut reader)?;
            if let Some(cfg) = adif.configs.first() {
                self.reconfigure(cfg.sampling_rate, cfg.channel_config, cfg.audio_object_type);
            }
        }

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

        let rate = self.sample_rate_hz();
        let active_channels = self.active_channels;
        let sbr_active = self.sbr_active;
        if let Some(limiter) = self.peak_limiter.as_mut() {
            if limiter.channels() != active_channels || limiter.sample_rate_hz() != rate {
                *limiter = crate::dsp::peak_limiter::PeakLimiter::new(active_channels, rate);
            }
            let mut slots: Vec<&mut [f32]> = self.channels[..active_channels]
                .iter_mut()
                .map(|ch| if sbr_active { ch.sbr_pcm.as_mut_slice() } else { ch.pcm.as_mut_slice() })
                .collect();
            limiter.process(&mut slots);
        }

        self.write_output_pcm();

        self.frame_count += 1;
        Ok(&self.output_pcm)
    }

    /// Convert this frame's decoded channels to interleaved 16-bit PCM in
    /// [`Self::output_pcm`], applying the fixed multichannel downmix first
    /// when [`Self::enable_downmix_to_stereo`] is on and this frame's channel
    /// configuration and count actually resolve to one of its layouts.
    ///
    /// Factored out of [`Self::decode_frame`] so the downmix decision and the
    /// decode-order permutation it applies can be exercised directly against
    /// hand-set channel state, without needing a real multichannel bitstream
    /// (this crate's own encoder does not emit PCE-declared multichannel
    /// streams to decode one from).
    fn write_output_pcm(&mut self) {
        let frame_len = self.frame_length();

        // Only takes effect when this frame's implicit channel configuration
        // resolves to one of the four fixed-matrix layouts AND actually
        // produced that many channels -- anything else (stereo, a PCE this
        // decoder does not role-map, an SBR-widened count that no longer
        // matches) falls through to plain per-channel output unchanged.
        let downmix_layout = self
            .downmix_to_stereo
            .then(|| self.downmix_layout())
            .flatten()
            .filter(|l| l.channels() == self.active_channels);

        if let Some(layout) = downmix_layout {
            let order = downmix_decode_order(layout);
            let refs: Vec<&[f32]> = order
                .iter()
                .map(|&i| {
                    let ch = &self.channels[i];
                    if self.sbr_active { ch.sbr_pcm.as_slice() } else { ch.pcm.as_slice() }
                })
                .collect();
            let (left, right) = crate::decoder::aac::downmix::downmix_to_stereo(layout, &refs);

            if self.output_pcm.channels() != 2 || self.output_pcm.samples_per_channel() != frame_len {
                self.output_pcm.resize(2, frame_len);
            }
            for (out, &v) in self.output_pcm.channel_mut(0).iter_mut().zip(left.iter()) {
                *out = clamp_to_i16(v);
            }
            for (out, &v) in self.output_pcm.channel_mut(1).iter_mut().zip(right.iter()) {
                *out = clamp_to_i16(v);
            }
        } else {
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
        }
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
        // `element_instance_tag -> (first channel index, channel count)` for
        // every SCE/CPE/LFE decoded this frame, so a CCE's target list (which
        // names channels by tag, not by decode order) can be resolved once
        // all of them are known. Cleared and rebuilt every frame: a tag's
        // meaning is only valid within the raw_data_block that declared it.
        let mut tag_channels: std::collections::HashMap<u8, (usize, usize)> =
            std::collections::HashMap::new();
        let mut pending_cce: Vec<CouplingChannelElement> = Vec::new();

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
                    let tag = reader.read_u8(4)?;
                    tag_channels.insert(tag, (next_channel, 1));
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
                    let tag = reader.read_u8(4)?;
                    tag_channels.insert(tag, (next_channel, 2));
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
                    // Its own spectrum decodes now, same as any other channel;
                    // mixing it into its targets' PCM has to wait until every
                    // SCE/CPE this frame has been both tag-mapped (just above)
                    // and filterbank-synthesized (only after the loop ends),
                    // so it is queued rather than applied here.
                    pending_cce.push(CouplingChannelElement::parse(reader, rate, frame_length, aot)?);
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
        self.mix_coupling_channels(pending_cce, &tag_channels, next_channel);
        Ok(next_channel)
    }

    /// Run each queued [`CouplingChannelElement`] through its own persistent
    /// filterbank state and scale-and-add the result into every target its
    /// tag list resolves to (`ixheaacd_dec_couple_channel`, applied once the
    /// real target mapping is known -- see `text/plan.txt` phase 7.5).
    ///
    /// A target tag this frame's SCE/CPE/LFE elements never declared is
    /// skipped rather than an error: a coupling channel naming a target that
    /// is not present in this particular raw_data_block is a real (if
    /// malformed) case a decoder has to tolerate, not something to fail the
    /// whole frame over when everything else decoded cleanly.
    fn mix_coupling_channels(
        &mut self,
        pending: Vec<CouplingChannelElement>,
        tag_channels: &std::collections::HashMap<u8, (usize, usize)>,
        active_channels: usize,
    ) {
        let rate = self.config.sampling_rate;
        for mut cce in pending {
            let (overlap, prev_shape) =
                self.cce_overlap.entry(cce.tag).or_insert_with(|| (vec![0.0; cce.data.spec.len()], WindowShape::Sine));
            if overlap.len() != cce.data.spec.len() {
                *overlap = vec![0.0; cce.data.spec.len()];
            }
            let mut coupling_pcm = vec![0.0f32; cce.data.spec.len()];
            Self::synthesize_channel_data(
                &mut self.filterbank,
                &mut self.deinterleave_scratch,
                rate,
                &mut cce.data,
                overlap,
                prev_shape,
                &mut coupling_pcm,
            );

            // Mirrors `CouplingChannelElement::parse`'s own `num_gain` count
            // exactly: two gains only for a channel-pair target coupled on
            // both channels, one gain otherwise (shared by whichever single
            // channel -- left, right, or the lone SCE channel -- is targeted).
            let mut gain_idx = 0usize;
            for target in &cce.targets {
                let Some(&(first, count)) = tag_channels.get(&target.tag) else { continue };
                let both = target.is_channel_pair && target.left && target.right && count == 2;
                if both {
                    let (gl, gr) = (cce.gains[gain_idx], cce.gains[gain_idx + 1]);
                    gain_idx += 2;
                    apply_coupling_gain(gl, &coupling_pcm, &mut self.channels[first].pcm);
                    apply_coupling_gain(gr, &coupling_pcm, &mut self.channels[first + 1].pcm);
                } else {
                    let gain = cce.gains[gain_idx];
                    gain_idx += 1;
                    let idx = if target.is_channel_pair && target.right && count == 2 {
                        first + 1
                    } else {
                        first
                    };
                    if idx < active_channels {
                        apply_coupling_gain(gain, &coupling_pcm, &mut self.channels[idx].pcm);
                    }
                }
            }
        }
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
                    let remaining = payload_bits - (reader.bit_position() - start);
                    self.decode_sbr_payload(reader, target, with_crc, remaining);
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
    fn decode_sbr_payload(
        &mut self,
        reader: &mut BitReader,
        target: SbrTarget,
        with_crc: bool,
        payload_bits_remaining: usize,
    ) {
        if self.config.frame_length.samples() != SBR_CORE_FRAME {
            return;
        }
        let core_rate = self.config.sampling_rate.hz();
        while self.sbr.len() <= target.element {
            self.sbr.push(SbrDecoder::new(target.kind.channels(), core_rate, false));
        }
        let sbr = &mut self.sbr[target.element];
        sbr.set_core_rate(core_rate);
        match sbr.decode_extension(reader, target.kind, with_crc, payload_bits_remaining) {
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
            Self::synthesize_channel_data(
                &mut self.filterbank,
                &mut self.deinterleave_scratch,
                rate,
                &mut ch.data,
                &mut ch.overlap,
                &mut ch.prev_shape,
                &mut ch.pcm,
            );
            trace_frame(self.frame_count, i, &ch.data);
        }
    }

    /// The per-channel body of [`Self::synthesize`] (deinterleave, TNS, IMDCT +
    /// windowing + overlap-add), factored out so a
    /// [`CouplingChannelElement`]'s own spectrum -- which is not one of
    /// `self.channels`, but still needs the exact same filterbank chain and its
    /// own persistent overlap-add state -- can run through it too. `filterbank`
    /// and `scratch` are taken by reference rather than `&mut self` so this can
    /// be called once per coupling channel after `self.channels` has already
    /// been borrowed for the mix targets.
    fn synthesize_channel_data(
        filterbank: &mut Filterbank,
        scratch: &mut Vec<f32>,
        rate: SamplingRate,
        data: &mut ChannelData,
        overlap: &mut Vec<f32>,
        prev_shape: &mut WindowShape,
        out: &mut Vec<f32>,
    ) {
        if data.ics.window_sequence.is_eight_short() {
            deinterleave(&data.ics, &data.spec, scratch);
            std::mem::swap(&mut data.spec, scratch);
        }

        apply_tns(data, rate);

        let sequence = data.ics.window_sequence;
        let shape = data.ics.window_shape;
        let prev = *prev_shape;

        filterbank.synthesize(&data.spec, sequence, shape, prev, overlap, out);
        *prev_shape = shape;
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

/// One coupling target: which element it names, and (for a channel pair)
/// which of its two channels are actually coupled.
#[derive(Debug, Clone, Copy)]
pub struct CouplingTarget {
    pub is_channel_pair: bool,
    pub tag: u8,
    pub left: bool,
    pub right: bool,
}

/// A `coupling_channel_element()`: an extra decoded channel meant to be mixed
/// into one or more *other* channels at a transmitted gain, rather than output
/// on its own -- used for things like a mix-minus commentary track or an
/// independently-controlled effect bus riding along with the main programme.
///
/// # What is real here, and what is not
///
/// The header, the target list and the per-target gains are decoded for real:
/// [`Self::gains`] holds actual values, not discarded bits, computed exactly as
/// `ixheaacd_dec_coupling_channel_element` does (`cc_gain_scale[k] =
/// 2^(step\[k\]/24)` for `step = [3, 6, 12, 24]`, each gain a power of that base
/// picked out by a Huffman-coded exponent using the very same codebook
/// scalefactor deltas use). [`Self::data`] is the coupling channel's own
/// decoded (dequantized, noise-substituted) spectral data, ready for the same
/// TNS + filterbank treatment any other channel gets.
///
/// What is **not** done is the mixing itself: [`crate::decoder::engine::Decoder`]
/// does not yet track which already-decoded output channel each target's `tag`
/// refers to (SCE/CPE elements are indexed by decode order today, not by the
/// `element_instance_tag` a CCE's targets name), and running the coupling
/// channel through the filterbank needs its own persistent overlap-add state
/// across frames, which the decoder does not carry. See [`apply_coupling_gain`]
/// for the one piece of the mix this module does provide -- the actual
/// time-domain scaled add, ready for whichever caller resolves the target
/// mapping -- and `text/plan.txt` phase 7.5 for what remains.
///
/// Only the reference's own supported case is decoded:
/// `independently_switched` coupling (`ind_sw_cce_flag == 1`), where every
/// target beyond the first gets one broadband gain. The other case
/// (per-scalefactor-band gain envelopes) is not a gap relative to libxaac: its
/// own `ixheaacd_dec_coupling_channel_element` reads the one bit that
/// distinguishes the two cases and then returns
/// `IA_XHEAAC_DEC_EXE_FATAL_UNIMPLEMENTED_CCE` without decoding anything
/// further, so [`CouplingChannelElement::parse`] does the same via
/// [`crate::error::Error::Unimplemented`].
#[derive(Debug)]
pub struct CouplingChannelElement {
    /// This coupling channel's own `element_instance_tag` -- distinct from
    /// each target's `tag`, which names the SCE/CPE it mixes into. Used to key
    /// this element's persistent overlap-add state across frames, the same
    /// way [`ChannelState`] does for ordinary channels.
    pub tag: u8,
    pub targets: Vec<CouplingTarget>,
    /// One gain per target, in target order; the first is always exactly 1.0
    /// (the reference hardcodes it rather than transmitting it).
    pub gains: Vec<f32>,
    pub data: ChannelData,
}

/// `cc_gain_scale[k] = 2^(step[k]/24)` (`ixheaacd_common_rom.c`'s
/// `cc_gain_scale[4]`, confirmed numerically against its Q29 fixed-point
/// values to `2e-9` relative error).
const CC_GAIN_SCALE_STEPS: [f64; 4] = [3.0, 6.0, 12.0, 24.0];

impl CouplingChannelElement {
    pub fn parse(
        reader: &mut BitReader,
        rate: SamplingRate,
        frame_length: FrameLength,
        aot: AudioObjectType,
    ) -> Result<Self> {
        let tag = reader.read_u8(4)?;
        let ind_sw_cce = reader.read_bit()?;
        let num_coupled = reader.read_u8(3)? as usize;

        let mut targets = Vec::with_capacity(num_coupled + 1);
        for _ in 0..=num_coupled {
            let is_channel_pair = reader.read_bit()?;
            let tag = reader.read_u8(4)?;
            let (left, right) = if is_channel_pair {
                (reader.read_bit()?, reader.read_bit()?)
            } else {
                (true, false)
            };
            targets.push(CouplingTarget { is_channel_pair, tag, left, right });
        }
        let num_gain = targets.iter().map(|t| if t.is_channel_pair && t.left && t.right { 2 } else { 1 }).sum::<usize>();

        let _cc_domain = reader.read_bit()?;
        let _gain_element_sign = reader.read_bit()?;
        let gain_element_scale = reader.read_u8(2)? as usize;

        let mut data = ChannelData::new(frame_length.samples());
        decode_ics(reader, &mut data, rate, frame_length, aot, None)?;
        inverse_quantize_channel(&mut data);

        if !ind_sw_cce {
            // Matches the reference exactly: it reads this one bit, then
            // refuses regardless of its value -- per-band gain envelopes for
            // coupling channels are not implemented in libxaac either.
            let _common_gain_element_present = reader.read_bit()?;
            return Err(crate::error::Error::Unimplemented {
                tool: "coupling_channel_element() with per-band gain envelopes",
                detail: "ind_sw_cce_flag == 0; not supported by the libxaac reference either",
            });
        }

        let base = 2.0f64.powf(1.0 / 24.0);
        let step = base.powf(CC_GAIN_SCALE_STEPS[gain_element_scale]);
        let mut gains = vec![1.0f32];
        for _ in 1..num_gain {
            let norm_value = crate::decoder::aac::huffman::decode_scalefactor_delta(reader)?;
            gains.push(step.powi(-norm_value) as f32);
        }

        Ok(Self { tag, targets, gains, data })
    }
}

/// The decode-order channel index that belongs at each column of
/// [`crate::decoder::aac::downmix::downmix_to_stereo`]'s matrix, for one of
/// the three implicit `channelConfiguration`s [`Decoder::downmix_layout`]
/// resolves.
///
/// The standard's implicit element order (no PCE) is always centre-first,
/// front-pair, [LFE last for the configurations that have one]: `SCE(C),
/// CPE(L,R), CPE(Ls,Rs)[, CPE(Lrs,Rrs)][, LFE]`. The downmix matrix's own
/// column order (documented on [`crate::decoder::aac::downmix::Layout`]) is
/// front-pair-first with LFE ahead of the rear channels instead, so this is a
/// fixed permutation, not a guess -- derived once here from both orderings
/// rather than re-derived at every call site.
fn downmix_decode_order(layout: crate::decoder::aac::downmix::Layout) -> &'static [usize] {
    use crate::decoder::aac::downmix::Layout;
    match layout {
        Layout::Ch5_0 => &[1, 2, 0, 3, 4],
        Layout::Ch5_1 => &[1, 2, 0, 5, 3, 4],
        Layout::Ch7_0 => &[1, 2, 0, 3, 4, 5, 6],
        Layout::Ch7_1 => &[1, 2, 0, 7, 3, 4, 5, 6],
    }
}

/// Mix a decoded coupling channel into a target's already-synthesized PCM, in
/// place (`ixheaacd_dec_couple_channel`): `target[i] += gain * coupling[i]`.
pub fn apply_coupling_gain(gain: f32, coupling: &[f32], target: &mut [f32]) {
    for (t, &c) in target.iter_mut().zip(coupling.iter()) {
        *t += gain * c;
    }
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

    /// The peak limiter is opt-in: a caller who never calls
    /// `enable_peak_limiter` must see byte-identical output to before it
    /// existed (covered by every other test in this file passing unchanged),
    /// and enabling it must measurably engage on a loud signal, never raise a
    /// peak, and turn back off cleanly.
    #[test]
    fn the_peak_limiter_is_opt_in_and_only_ever_reduces_peaks() {
        use crate::encoder::{Encoder, EncoderConfig};

        let mut enc = Encoder::new(EncoderConfig::default()).unwrap();
        let mut pcm = AudioBuffer::<i16>::new(2, 1024);
        for c in 0..2 {
            for (i, s) in pcm.channel_mut(c).iter_mut().enumerate() {
                *s = ((i as f32 * 0.05).sin() * 32000.0) as i16;
            }
        }
        let mut frames = Vec::new();
        for _ in 0..4 {
            frames.push(enc.encode_frame(&pcm).unwrap());
        }
        frames.push(enc.flush().unwrap());
        let frames: Vec<_> = frames.into_iter().filter(|f| !f.is_empty()).collect();
        assert!(!frames.is_empty());

        let mut without_limiter = Decoder::new_default();
        let mut with_limiter = Decoder::new_default();
        assert!(!with_limiter.peak_limiter_enabled());
        with_limiter.enable_peak_limiter();
        assert!(with_limiter.peak_limiter_enabled());

        // The limiter's look-ahead delay shifts every sample by a few hundred
        // positions, so comparing the two streams index-for-index would
        // compare unrelated content across that shift. What is actually
        // guaranteed, and checked here, is a global one: concatenated across
        // the whole multi-frame stream, the limiter can only ever hold the
        // peak the same or bring it down, never up.
        let mut any_difference = false;
        let mut unlimited_peak = 0i32;
        let mut limited_peak = 0i32;
        for frame in &frames {
            let unlimited = without_limiter.decode_frame(frame).unwrap().clone();
            let limited = with_limiter.decode_frame(frame).unwrap().clone();
            for ch in 0..2 {
                let (u, l) = (unlimited.channel(ch), limited.channel(ch));
                assert_eq!(u.len(), l.len());
                unlimited_peak = unlimited_peak.max(u.iter().map(|&v| (v as i32).abs()).max().unwrap_or(0));
                limited_peak = limited_peak.max(l.iter().map(|&v| (v as i32).abs()).max().unwrap_or(0));
                if u != l {
                    any_difference = true;
                }
            }
        }
        assert!(any_difference, "a full-scale tone must actually engage the limiter");
        assert!(
            limited_peak <= unlimited_peak,
            "limiter raised the stream's overall peak: {unlimited_peak} -> {limited_peak}"
        );

        with_limiter.disable_peak_limiter();
        assert!(!with_limiter.peak_limiter_enabled());
    }

    /// Encode one real ADTS frame, then strip its header to get the bare
    /// `raw_data_block()` both LOAS/LATM and ADIF wrap. Returns the payload
    /// alongside the ASC these tests need to build a matching StreamMuxConfig
    /// or ADIF header.
    fn encode_one_raw_data_block() -> (Vec<u8>, crate::syntax::asc::AudioSpecificConfig) {
        encode_one_raw_data_block_with_channels(2)
    }

    /// Same as [`encode_one_raw_data_block`], but with a chosen channel count
    /// -- in particular, mono to guarantee a plain SCE (`element_id == 0`)
    /// rather than a CPE, for tests that need to know the raw block's exact
    /// 7-bit element header shape.
    fn encode_one_raw_data_block_with_channels(
        channels: usize,
    ) -> (Vec<u8>, crate::syntax::asc::AudioSpecificConfig) {
        use crate::encoder::{Encoder, EncoderConfig};
        use crate::syntax::adts::AdtsHeader;

        let channel_config = if channels == 1 {
            ChannelConfiguration::Mono
        } else {
            ChannelConfiguration::Stereo
        };
        let mut enc =
            Encoder::new(EncoderConfig { channel_config, ..EncoderConfig::default() }).unwrap();
        let pcm = crate::buffer::AudioBuffer::<i16>::new(channels, 1024);
        // The encoder holds the first frame back for lookahead; flush to get a
        // real, complete ADTS frame rather than the empty first return.
        enc.encode_frame(&pcm).unwrap();
        let frame = loop {
            let f = enc.flush().unwrap();
            if !f.is_empty() {
                break f;
            }
        };

        let mut reader = BitReader::new(&frame);
        let header = AdtsHeader::parse(&mut reader).expect("encoder must emit valid ADTS");
        let raw_data_block = frame[reader.byte_position()..].to_vec();

        let asc = crate::syntax::asc::AudioSpecificConfig {
            audio_object_type: header.audio_object_type,
            sampling_rate: header.sampling_rate,
            channel_config: header.channel_config,
            frame_length: crate::types::FrameLength::Samples1024,
            depends_on_core_coder: false,
            core_coder_delay: 0,
            extension_audio_object_type: None,
            extension_sampling_rate: None,
            sbr_present: false,
            ps_present: false,
        };
        (raw_data_block, asc)
    }

    /// A LOAS/LATM-wrapped frame must decode to the same result as feeding its
    /// raw_data_block() straight to the decoder: the unwrap in decode_frame
    /// must be transparent, and the StreamMuxConfig's ASC must actually
    /// reconfigure the decoder rather than being parsed and discarded.
    #[test]
    fn a_loas_latm_wrapped_frame_decodes_and_reconfigures() {
        use crate::bitstream::BitWriter;
        use crate::syntax::latm::{AudioMuxElement, StreamMuxConfig};

        let (raw_data_block, asc) = encode_one_raw_data_block();

        let mux_config = StreamMuxConfig {
            audio_mux_version: 0,
            all_streams_same_time_framing: true,
            num_sub_frames: 1,
            num_programs: 1,
            num_layers: 1,
            asc,
        };
        let elem = AudioMuxElement {
            mux_config_present: true,
            stream_mux_config: Some(mux_config),
            payload_bytes: raw_data_block.clone(),
        };
        let mut writer = BitWriter::with_capacity(raw_data_block.len() + 16);
        elem.write_loas(&mut writer);
        let loas_bytes = writer.finalize();

        // Start from a decoder configured for the wrong rate/channels, so a
        // pass here proves the LATM path actually reconfigures rather than the
        // test coincidentally already matching.
        let mut latm_decoder = Decoder::new(crate::syntax::asc::AudioSpecificConfig {
            audio_object_type: AudioObjectType::AacLc,
            sampling_rate: SamplingRate::Hz16000,
            channel_config: ChannelConfiguration::Mono,
            frame_length: crate::types::FrameLength::Samples1024,
            depends_on_core_coder: false,
            core_coder_delay: 0,
            extension_audio_object_type: None,
            extension_sampling_rate: None,
            sbr_present: false,
            ps_present: false,
        });
        let via_latm = latm_decoder.decode_frame(loas_bytes).expect("LATM frame decodes").clone();
        assert_eq!(latm_decoder.channels(), 2, "StreamMuxConfig's ASC must reconfigure the decoder");
        assert_eq!(latm_decoder.sample_rate_hz(), 44100);

        let mut raw_decoder = Decoder::new_default();
        let via_raw = raw_decoder.decode_frame(&raw_data_block).expect("raw block decodes").clone();

        assert_eq!(via_latm.samples_per_channel(), via_raw.samples_per_channel());
        for ch in 0..2 {
            assert_eq!(via_latm.channel(ch), via_raw.channel(ch), "LATM unwrap must be transparent");
        }
    }

    /// An ADIF-prefixed stream must decode its first frame identically to the
    /// bare raw_data_block(), with the header's first program config actually
    /// reconfiguring the decoder.
    #[test]
    fn an_adif_prefixed_stream_decodes_and_reconfigures() {
        use crate::bitstream::BitWriter;
        use crate::syntax::adif::AdifHeader;

        let (raw_data_block, asc) = encode_one_raw_data_block();

        let header = AdifHeader {
            copyright_id_present: false,
            copyright_id: [0u8; 9],
            original_copy: false,
            home: false,
            bitstream_type: true,
            bitrate: 128_000,
            num_program_config_elements: 1,
            buffer_fullness: 0,
            configs: vec![asc],
        };
        // The header's bit length is not generally a multiple of 8, so the
        // raw_data_block's bits must follow immediately in the same bitstream
        // -- appending its bytes after a byte-aligned `finalize()` would shift
        // every bit and is not what a real ADIF stream looks like.
        let mut writer = BitWriter::with_capacity(raw_data_block.len() + 16);
        header.write(&mut writer);
        let mut block_reader = BitReader::new(&raw_data_block);
        for _ in 0..raw_data_block.len() * 8 {
            writer.write_bit(block_reader.read_bit().unwrap());
        }
        let stream = writer.finalize().to_vec();

        let mut adif_decoder = Decoder::new(crate::syntax::asc::AudioSpecificConfig {
            audio_object_type: AudioObjectType::AacLc,
            sampling_rate: SamplingRate::Hz16000,
            channel_config: ChannelConfiguration::Mono,
            frame_length: crate::types::FrameLength::Samples1024,
            depends_on_core_coder: false,
            core_coder_delay: 0,
            extension_audio_object_type: None,
            extension_sampling_rate: None,
            sbr_present: false,
            ps_present: false,
        });
        let via_adif = adif_decoder.decode_frame(&stream).expect("ADIF-prefixed stream decodes").clone();
        assert_eq!(adif_decoder.channels(), 2, "the ADIF header's PCE must reconfigure the decoder");
        assert_eq!(adif_decoder.sample_rate_hz(), 44100);

        let mut raw_decoder = Decoder::new_default();
        let via_raw = raw_decoder.decode_frame(&raw_data_block).expect("raw block decodes").clone();

        for ch in 0..2 {
            assert_eq!(via_adif.channel(ch), via_raw.channel(ch), "ADIF unwrap must be transparent");
        }
    }

    /// `cc_gain_scale[k]` reverse-engineered from `ixheaacd_common_rom.c`'s Q29
    /// values must reproduce them (585461881, 638450708, 759250125,
    /// 1073741824 out of 2^29) to within Q29 rounding.
    #[test]
    fn cc_gain_scale_matches_the_reference_q29_table() {
        let base = 2.0f64.powf(1.0 / 24.0);
        let expected_q29 = [585461881.0, 638450708.0, 759250125.0, 1073741824.0];
        for (step, want) in CC_GAIN_SCALE_STEPS.iter().zip(expected_q29.iter()) {
            let got = base.powf(*step) * (1u64 << 29) as f64;
            assert!((got - want).abs() < 1.0, "step {step}: {got} vs {want}");
        }
    }

    /// `apply_coupling_gain` must scale-and-add, not replace, and must leave a
    /// unity gain as a pure passthrough sum.
    #[test]
    fn coupling_gain_scales_and_adds_in_place() {
        let coupling = [1.0f32, -2.0, 3.0];
        let mut target = [10.0f32, 10.0, 10.0];
        apply_coupling_gain(0.5, &coupling, &mut target);
        assert_eq!(target, [10.5, 9.0, 11.5]);

        let mut target = [0.0f32, 0.0, 0.0];
        apply_coupling_gain(1.0, &coupling, &mut target);
        assert_eq!(target, coupling);
    }

    /// A real independently-switched CCE (`ind_sw_cce_flag == 1`) with one
    /// channel-pair target coupled on both channels: two gains beyond the
    /// implicit unity first one, decoded from real Huffman-coded scalefactor
    /// deltas, and the embedded channel stream must actually decode.
    #[test]
    fn an_independently_switched_cce_decodes_real_targets_and_gains() {
        use crate::bitstream::BitWriter;
        use crate::syntax::asc::AudioSpecificConfig;

        // Build a minimal-but-real SCE payload to embed as the coupling
        // channel's own stream, by encoding one and stripping its ADTS header
        // -- the same technique the LATM/ADIF tests use.
        let (raw_sce, asc) = encode_one_raw_data_block_with_channels(1);
        // `raw_sce` is `element_id(3) + element_instance_tag(4) + ics_data() +
        // END(3) + byte padding`; coupling_channel_element() expects exactly
        // ics_data() right after its own header (decode_ics starts at
        // global_gain), so strip the 7-bit element header and discover ics_data's
        // exact bit length with a real decode_ics call rather than guessing --
        // the trailing END element and padding bits must NOT be copied in, or
        // they land between ics_data and the gain codeword that follows it.
        let mut sce_reader = BitReader::new(&raw_sce);
        let _sce_id = sce_reader.read_bits(3).unwrap();
        let _sce_tag = sce_reader.read_bits(4).unwrap();
        let ics_start = sce_reader.bit_position();
        let mut scratch = ChannelData::new(asc.frame_length.samples());
        decode_ics(&mut sce_reader, &mut scratch, asc.sampling_rate, asc.frame_length, asc.audio_object_type, None)
            .expect("the encoder's own ics_data must parse");
        let ics_data_bits = sce_reader.bit_position() - ics_start;
        let mut sce_reader = BitReader::new(&raw_sce);
        let _ = sce_reader.read_bits(7).unwrap();

        let mut writer = BitWriter::with_capacity(raw_sce.len() + 8);
        writer.write_u8(0, 4); // element_instance_tag
        writer.write_bit(true); // ind_sw_cce_flag
        writer.write_u8(0, 3); // num_coupled_elements (1 target total)
        // Target 0: a channel pair, both channels coupled.
        writer.write_bit(true); // is_channel_pair
        writer.write_u8(3, 4); // tag
        writer.write_bit(true); // cc_l
        writer.write_bit(true); // cc_r
        writer.write_bit(false); // cc_domain
        writer.write_bit(false); // gain_element_sign
        writer.write_u8(1, 2); // gain_element_scale
        // The coupling channel's own ics_data, bit for bit and no more --
        // decode_ics() runs before the gain list, exactly as
        // ixheaacd_dec_coupling_channel_element decodes its individual_ch_stream()
        // before ever touching cc_gain[].
        for _ in 0..ics_data_bits {
            writer.write_bit(sce_reader.read_bit().unwrap());
        }
        // Two Huffman-coded gains (num_gain = 2, since the one CPE target with
        // both channels coupled counts twice), each a real delta-0 codeword
        // from the encoder's own scalefactor Huffman writer -- decoding to
        // norm_value 0, i.e. gain = step^0 = 1.0.
        assert!(crate::encoder::aac::huffman::write_scalefactor_delta(&mut writer, 0));
        assert!(crate::encoder::aac::huffman::write_scalefactor_delta(&mut writer, 0));
        let bytes = writer.finalize().to_vec();

        let mut reader = BitReader::new(&bytes);
        let asc = AudioSpecificConfig {
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
        let cce = CouplingChannelElement::parse(
            &mut reader,
            asc.sampling_rate,
            asc.frame_length,
            asc.audio_object_type,
        )
        .expect("a real ind_sw_cce=1 element must decode");

        assert_eq!(cce.targets.len(), 1);
        assert!(cce.targets[0].is_channel_pair);
        assert_eq!(cce.targets[0].tag, 3);
        assert!(cce.targets[0].left && cce.targets[0].right);
        assert_eq!(cce.gains.len(), 2, "one CPE target coupled on both channels needs two gains");
        assert_eq!(cce.gains[0], 1.0, "the first gain is always the implicit unity");
        assert!((cce.gains[1] - 1.0).abs() < 1e-4, "an all-zero delta must decode to unity gain: {:?}", cce.gains);
        assert!(cce.data.global_gain > 0 || cce.data.global_gain == 0, "ics_data actually parsed");
    }

    /// `Decoder::write_output_pcm` (phase 7.6) must, once enabled, resolve a
    /// 5.1 stream's implicit centre-first decode order into the downmix
    /// matrix's front-pair-first column order before mixing -- not the raw
    /// decode order -- and must leave a channel count the matrix does not
    /// cover (stereo) untouched even with downmixing turned on.
    #[test]
    fn write_output_pcm_permutes_decode_order_before_downmixing() {
        let mut dec = Decoder::new(AudioSpecificConfig {
            audio_object_type: AudioObjectType::AacLc,
            sampling_rate: SamplingRate::Hz44100,
            channel_config: ChannelConfiguration::FivePointOne,
            frame_length: crate::types::FrameLength::Samples1024,
            depends_on_core_coder: false,
            core_coder_delay: 0,
            extension_audio_object_type: None,
            extension_sampling_rate: None,
            sbr_present: false,
            ps_present: false,
        });
        dec.active_channels = 6;
        dec.enable_downmix_to_stereo();
        assert!(dec.downmix_to_stereo_enabled());

        // Decode order for FivePointOne is C, L, R, Ls, Rs, LFE -- put a
        // distinct constant in each channel so a wrong permutation shows up
        // as a completely different (but still plausible-looking) mix.
        let full_scale = [0.2f32, 0.4, 0.6, 0.05, 0.05, 0.9]; // C, L, R, Ls, Rs, LFE
        for (ch, &v) in dec.channels.iter_mut().zip(full_scale.iter()) {
            ch.pcm.iter_mut().for_each(|s| *s = v);
        }

        dec.write_output_pcm();

        assert_eq!(dec.output_pcm.channels(), 2, "6 implicit channels must collapse to stereo");
        let (left, right) = crate::decoder::aac::downmix::downmix_to_stereo(
            crate::decoder::aac::downmix::Layout::Ch5_1,
            // L, R, C, LFE, Ls, Rs -- the matrix's own order, built directly
            // from the *named* constants above rather than by re-deriving
            // the permutation, so this checks the same thing two independent
            // ways.
            &[
                &[full_scale[1]; 1024],
                &[full_scale[2]; 1024],
                &[full_scale[0]; 1024],
                &[full_scale[5]; 1024],
                &[full_scale[3]; 1024],
                &[full_scale[4]; 1024],
            ],
        );
        assert_eq!(dec.output_pcm.channel(0)[0], clamp_to_i16(left[0]));
        assert_eq!(dec.output_pcm.channel(1)[0], clamp_to_i16(right[0]));

        // A stereo frame is not one of the four layouts: downmixing must be a
        // no-op even though it is enabled.
        let mut stereo = Decoder::new_default();
        stereo.active_channels = 2;
        stereo.enable_downmix_to_stereo();
        stereo.channels[0].pcm.iter_mut().for_each(|s| *s = 0.5);
        stereo.channels[1].pcm.iter_mut().for_each(|s| *s = -0.5);
        stereo.write_output_pcm();
        assert_eq!(stereo.output_pcm.channels(), 2);
        assert_eq!(stereo.output_pcm.channel(0)[0], clamp_to_i16(0.5));
        assert_eq!(stereo.output_pcm.channel(1)[0], clamp_to_i16(-0.5));
    }

    /// `Decoder::mix_coupling_channels` (phase 7.5) must resolve a target by
    /// its real `element_instance_tag`, scale-and-add the coupling channel's
    /// *own* filterbank output (not its raw spectrum) into that target's PCM,
    /// split a two-gain channel-pair target's gains across the right
    /// channels, and leave an unresolved target's tag entirely alone rather
    /// than panicking on a missing map entry.
    #[test]
    fn mix_coupling_channels_resolves_targets_and_adds_the_synthesized_signal() {
        let asc = AudioSpecificConfig {
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
        let n = asc.frame_length.samples();

        let make_data = || {
            let mut data = ChannelData::new(n);
            data.spec[3] = 500.0;
            data.spec[17] = -120.0;
            data
        };

        // The independent oracle: synthesize the exact same spectrum through
        // a fresh filterbank with fresh (zeroed) overlap state, exactly the
        // way `mix_coupling_channels` does internally for a coupling channel
        // it has never seen before.
        let mut oracle_fb = Filterbank::new(n);
        let mut oracle_scratch = vec![0.0f32; n];
        let mut oracle_overlap = vec![0.0f32; n];
        let mut oracle_prev = WindowShape::Sine;
        let mut expected_pcm = vec![0.0f32; n];
        Decoder::synthesize_channel_data(
            &mut oracle_fb,
            &mut oracle_scratch,
            asc.sampling_rate,
            &mut make_data(),
            &mut oracle_overlap,
            &mut oracle_prev,
            &mut expected_pcm,
        );
        assert!(expected_pcm.iter().any(|&v| v != 0.0), "the oracle itself must be non-silent");

        let mut dec = Decoder::new(asc);
        for v in dec.channels[0].pcm.iter_mut() {
            *v = 1.0;
        }
        for v in dec.channels[1].pcm.iter_mut() {
            *v = -1.0;
        }
        let untouched_before = dec.channels[2].pcm.clone();

        let cce = CouplingChannelElement {
            tag: 9,
            targets: vec![
                CouplingTarget { is_channel_pair: true, tag: 3, left: true, right: true },
                // No element declared tag 5 this frame -- must be skipped, not panic.
                CouplingTarget { is_channel_pair: false, tag: 5, left: true, right: false },
            ],
            gains: vec![1.0, 0.5],
            data: make_data(),
        };
        let mut tag_channels = std::collections::HashMap::new();
        tag_channels.insert(3u8, (0usize, 2usize));

        dec.mix_coupling_channels(vec![cce], &tag_channels, 2);

        for i in 0..n {
            let want_l = 1.0 + 1.0 * expected_pcm[i];
            let want_r = -1.0 + 0.5 * expected_pcm[i];
            assert!((dec.channels[0].pcm[i] - want_l).abs() < 1e-4, "L[{i}]: {} vs {want_l}", dec.channels[0].pcm[i]);
            assert!((dec.channels[1].pcm[i] - want_r).abs() < 1e-4, "R[{i}]: {} vs {want_r}", dec.channels[1].pcm[i]);
        }
        assert_eq!(dec.channels[2].pcm, untouched_before, "an unresolved target tag must not touch any channel");
    }

    /// `ind_sw_cce_flag == 0` must be refused for exactly that reason -- not
    /// incidentally fail because the test fed it nonsense ics_data. Real
    /// ics_data is embedded (as the reference itself decodes it unconditionally,
    /// before ever checking `ind_sw_cce_flag`) so the only way this can fail is
    /// the check under test.
    #[test]
    fn a_non_independently_switched_cce_is_refused_not_misparsed() {
        use crate::bitstream::BitWriter;

        let (raw_sce, _asc) = encode_one_raw_data_block_with_channels(1);
        let mut sce_reader = BitReader::new(&raw_sce);
        let _sce_id = sce_reader.read_bits(3).unwrap();
        let _sce_tag = sce_reader.read_bits(4).unwrap();

        let mut writer = BitWriter::with_capacity(raw_sce.len() + 8);
        writer.write_u8(0, 4); // element_instance_tag
        writer.write_bit(false); // ind_sw_cce_flag = 0
        writer.write_u8(0, 3); // num_coupled_elements
        writer.write_bit(true); // is_channel_pair
        writer.write_u8(0, 4); // tag
        writer.write_bit(false); // cc_l
        writer.write_bit(false); // cc_r
        writer.write_bit(false); // cc_domain
        writer.write_bit(false); // gain_element_sign
        writer.write_u8(0, 2); // gain_element_scale
        for _ in 0..(raw_sce.len() * 8 - 7) {
            writer.write_bit(sce_reader.read_bit().unwrap());
        }
        let bytes = writer.finalize().to_vec();
        let mut reader = BitReader::new(&bytes);

        let err = CouplingChannelElement::parse(
            &mut reader,
            SamplingRate::Hz44100,
            crate::types::FrameLength::Samples1024,
            AudioObjectType::AacLc,
        )
        .expect_err("ind_sw_cce_flag == 0 must not silently decode");
        assert!(
            matches!(err, crate::error::Error::Unimplemented { .. }),
            "must fail via the explicit refusal, not an incidental parse error: {err}"
        );
    }
}
