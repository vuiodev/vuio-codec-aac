//! AAC-LC encoder.
//!
//! Each frame windows the current input together with the previous frame's samples,
//! transforms them with the MDCT, quantizes the result under a bit budget, and
//! writes an ADTS frame.
//!
//! # Scope
//!
//! This encoder emits long windows with a sine shape. A masking model
//! ([`crate::encoder::aac::psycho`]) decides how much noise each band may carry, a
//! rate loop ([`crate::encoder::aac::rate`]) turns that into per-band scalefactors
//! that fit the frame's budget, and a stereo pair is coded mid/side wherever that
//! is cheaper. Block switching for transients and temporal noise shaping are not
//! wired in yet.

use crate::bitstream::{BitReader, BitWriter};
use crate::buffer::AudioBuffer;
use crate::dsp::fft::Complex32;
use crate::dsp::mdct::MdctContext;
use crate::dsp::filterbank::{frame_window, short_window, short_window_offset};
use crate::encoder::aac::huffman::write_scalefactor_delta;
use crate::encoder::aac::block_switch::{BlockDecision, BlockSwitch, SUB_BLOCKS, Transient};
use crate::encoder::aac::psycho::{MAX_BANDS, PsychoResult, PsychoacousticModel};
use crate::encoder::aac::quant::{SF_OFFSET, write_band};
use crate::encoder::aac::rate::{Quantization, RateLoop};
use crate::encoder::aac::tns::{TnsFilter, apply as apply_tns};
use crate::error::Result;
use crate::syntax::adif::AdifHeader;
use crate::syntax::adts::AdtsHeader;
use crate::syntax::asc::AudioSpecificConfig;
use crate::tables::scalefactor::{MAX_SFB_LONG, compute_sfb_offsets, get_sfb_table};
use crate::types::{
    AudioObjectType, ChannelConfiguration, FrameLength, SamplingRate, WindowSequence, WindowShape,
};

/// Which container framing [`Encoder`] wraps each `raw_data_block()` in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// A self-framing header on every frame -- the default, and what every
    /// existing caller of this encoder already gets.
    #[default]
    Adts,
    /// One header before the very first frame (carrying this stream's
    /// [`AudioSpecificConfig`] as its single program config element,
    /// [`crate::syntax::adif::AdifHeader`]'s own convention -- see that
    /// module's docs), then bare `raw_data_block()`s with no per-frame
    /// framing at all -- the first joined to the header bit for bit, since
    /// the header's own length is not generally byte-aligned.
    ///
    /// Each call to [`Encoder::encode_frame`]/[`Encoder::flush`] still
    /// returns one frame's bytes at a time (the first carrying the header, so
    /// it is larger than the rest), exactly like [`OutputFormat::Adts`]; feed
    /// each returned chunk to [`crate::decoder::engine::Decoder::decode_frame`]
    /// in turn (it already reads this shape end to end, phase 0.5) and it
    /// round-trips. Concatenating every chunk into one continuous file also
    /// produces a real, spec-shaped ADIF stream for other decoders -- but
    /// note that a *continuous, un-chunked* read of that file is not
    /// something `Decoder::decode_frame` supports today: each call still
    /// consumes exactly one `raw_data_block()` from the bytes it is given,
    /// never "the rest of the reader".
    Adif,
}

/// Encoder configuration.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub audio_object_type: AudioObjectType,
    pub sampling_rate: SamplingRate,
    pub channel_config: ChannelConfiguration,
    pub bitrate_bps: u32,
    pub frame_length: FrameLength,
    pub output_format: OutputFormat,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            audio_object_type: AudioObjectType::AacLc,
            sampling_rate: SamplingRate::Hz44100,
            channel_config: ChannelConfiguration::Stereo,
            bitrate_bps: 128_000,
            frame_length: FrameLength::Samples1024,
            output_format: OutputFormat::default(),
        }
    }
}

/// Per-channel encoder state.
#[derive(Debug, Clone)]
struct ChannelState {
    /// Previous frame's input, forming the first half of this frame's window.
    history: Vec<f32>,
    /// `2n` windowed samples.
    windowed: Vec<f32>,
    /// `n` spectral coefficients.
    spectrum: Vec<f32>,
    /// The masking model's view of this channel's frame.
    psycho: PsychoResult,
    /// What the rate loop settled on.
    coded: Quantization,
    /// The masking model for long windows, which carries pre-echo state between
    /// frames.
    model: PsychoacousticModel,
    /// The same for the eight-short frames, whose band table is a different one.
    short_model: PsychoacousticModel,
    /// Spectrum rearranged into the order the bitstream carries, which for an
    /// eight-short frame interleaves the windows of a group band by band.
    grouped: Vec<f32>,
    /// Noise shaping applied to this frame, if any.
    tns: Option<TnsFilter>,
    /// Scalefactor estimation and the search that fits the budget.
    ///
    /// One per channel rather than one per encoder: it carries the threshold scale
    /// the channel settled on last frame, and holding it here is also what lets the
    /// channels be quantized in parallel.
    rate: RateLoop,
}

impl ChannelState {
    fn new(
        n: usize,
        bands: usize,
        sample_rate_hz: u32,
        bitrate_per_channel: u32,
        offsets: &[usize],
        short_offsets: &[usize],
    ) -> Self {
        Self {
            history: vec![0.0; n],
            windowed: vec![0.0; 2 * n],
            spectrum: vec![0.0; n],
            psycho: PsychoResult::default(),
            coded: Quantization::new(n, bands),
            model: PsychoacousticModel::new(sample_rate_hz, bitrate_per_channel, offsets, false),
            short_model: PsychoacousticModel::new(
                sample_rate_hz,
                bitrate_per_channel,
                short_offsets,
                true,
            ),
            grouped: vec![0.0; n],
            tns: None,
            rate: RateLoop::new(n),
        }
    }
}

/// Bands the four-bit `max_sfb` field can name for a short window.
const MAX_SHORT_BANDS: usize = 15;

/// AAC-LC encoder.
#[derive(Debug, Clone)]
pub struct Encoder {
    config: EncoderConfig,
    mdct: MdctContext,
    /// Transform for the eight short windows of a transient frame.
    mdct_short: MdctContext,
    channels: Vec<ChannelState>,
    /// Cumulative band offsets for long windows, `num_bands + 1` entries.
    sfb_offsets: [usize; MAX_SFB_LONG + 1],
    /// The same for one short window.
    short_offsets: [usize; MAX_SFB_LONG + 1],
    num_bands: usize,
    /// Bands one short window has.
    short_bands: usize,
    /// Highest band the encoder codes with long windows.
    max_sfb: usize,
    /// The same for short windows.
    short_max_sfb: usize,
    /// Bit budget for one frame, from the requested bitrate.
    frame_bits: usize,
    frame_count: u64,
    mdct_scratch: Vec<Complex32>,
    writer: BitWriter,
    /// Bands coded mid/side, when the frame is a stereo pair.
    ms_mask: Vec<bool>,
    /// Whether any band of the current frame is coded mid/side.
    ms_used: bool,
    /// Transient detection, shared so that every channel of a frame agrees on the
    /// window sequence the element header carries.
    block_switch: BlockSwitch,
    /// The frame held back so the detector can see one frame further than it codes,
    /// and what it found there.
    pending: Option<(Vec<Vec<f32>>, Transient)>,
    /// Sequence and grouping the frame being coded uses.
    decision: BlockDecision,
    /// Band table the rate loop works from, which for an eight-short frame is the
    /// short table repeated once per window group.
    coding_offsets: [usize; MAX_BANDS + 1],
    /// Bands in [`Self::coding_offsets`].
    coding_bands: usize,
    /// Set once the one-time ADIF header has been emitted, in
    /// [`OutputFormat::Adif`]; irrelevant otherwise.
    adif_header_written: bool,
}

impl Encoder {
    /// Create an encoder.
    pub fn new(config: EncoderConfig) -> Result<Self> {
        let n = config.frame_length.samples();
        let num_ch = config.channel_config.channels().max(1);

        let widths = get_sfb_table(config.sampling_rate, false, config.frame_length);
        let mut sfb_offsets = [0usize; MAX_SFB_LONG + 1];
        let count = compute_sfb_offsets(widths, &mut sfb_offsets);
        let num_bands = count - 1;

        // Code the whole band table; the rate loop decides what survives.
        let max_sfb = num_bands.min(MAX_SFB_LONG);

        let frames_per_sec = config.sampling_rate.hz() as f64 / n as f64;
        let frame_bits = (config.bitrate_bps as f64 / frames_per_sec) as usize;

        let short_widths = get_sfb_table(config.sampling_rate, true, config.frame_length);
        let mut short_offsets = [0usize; MAX_SFB_LONG + 1];
        let short_count = compute_sfb_offsets(short_widths, &mut short_offsets);
        let short_bands = short_count - 1;

        let mdct = MdctContext::new(n);
        let mdct_short = MdctContext::new(n / SUB_BLOCKS);
        let scratch_len = mdct.scratch_len();
        let config_rate = config.sampling_rate.hz();
        let per_channel_bitrate = config.bitrate_bps / num_ch as u32;

        let mut coding_offsets = [0usize; MAX_BANDS + 1];
        coding_offsets[..=num_bands.min(MAX_BANDS)]
            .copy_from_slice(&sfb_offsets[..=num_bands.min(MAX_BANDS)]);

        Ok(Self {
            config,
            mdct,
            mdct_short,
            channels: (0..num_ch)
                .map(|_| {
                    ChannelState::new(
                        n,
                        MAX_BANDS,
                        config_rate,
                        per_channel_bitrate,
                        &sfb_offsets[..count],
                        &short_offsets[..short_count],
                    )
                })
                .collect(),
            sfb_offsets,
            short_offsets,
            num_bands,
            short_bands,
            max_sfb,
            short_max_sfb: short_bands.min(MAX_SHORT_BANDS),
            frame_bits,
            frame_count: 0,
            mdct_scratch: vec![Complex32::default(); scratch_len],
            writer: BitWriter::with_capacity(4096),
            ms_mask: vec![false; MAX_BANDS],
            ms_used: false,
            block_switch: BlockSwitch::new(per_channel_bitrate),
            pending: None,
            decision: BlockDecision {
                sequence: WindowSequence::OnlyLongSequence,
                groups: [1; SUB_BLOCKS],
                group_count: 1,
            },
            coding_offsets,
            coding_bands: num_bands.min(MAX_BANDS),
            adif_header_written: false,
        })
    }

    /// Frames encoded so far.
    #[inline]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Bits available for one frame's payload.
    #[inline]
    pub fn frame_bits(&self) -> usize {
        self.frame_bits
    }

    /// Reset inter-frame state.
    pub fn reset(&mut self) {
        for ch in self.channels.iter_mut() {
            ch.history.fill(0.0);
            ch.model.reset();
            ch.short_model.reset();
        }
        self.block_switch.reset();
        self.pending = None;
        self.frame_count = 0;
    }

    /// Encode one frame of PCM.
    ///
    /// The transient detector has to see one frame further than the encoder codes,
    /// because a frame of short windows must be announced by a start window in the
    /// frame before it. Each call therefore returns the frame *before* the one it is
    /// given, and the first call returns nothing; [`Self::flush`] emits the last
    /// frame once the input has run out.
    pub fn encode_frame(&mut self, pcm: &AudioBuffer<i16>) -> Result<Vec<u8>> {
        let num_ch = self.channels.len();
        let n = self.config.frame_length.samples();
        assert_eq!(pcm.channels(), num_ch, "input channel count mismatch");
        assert_eq!(pcm.samples_per_channel(), n, "input frame length mismatch");

        let samples: Vec<Vec<f32>> =
            (0..num_ch).map(|c| pcm.channel(c).iter().map(|&v| v as f32).collect()).collect();
        let transient = self.detect(&samples);

        match self.pending.replace((samples, transient)) {
            Some((frame, here)) => self.encode_pending(&frame, here, transient),
            None => Ok(Vec::new()),
        }
    }

    /// Emit the frame held back by the lookahead, if there is one.
    ///
    /// Call once after the last [`Self::encode_frame`]; calling it again returns
    /// nothing.
    pub fn flush(&mut self) -> Result<Vec<u8>> {
        match self.pending.take() {
            Some((frame, here)) => {
                let quiet = Transient { attack: false, sub_block: 0 };
                self.encode_pending(&frame, here, quiet)
            }
            None => Ok(Vec::new()),
        }
    }

    /// Look for a transient in one frame of input.
    ///
    /// One decision for the whole frame rather than one per channel: a channel pair
    /// shares its `ics_info`, so the two have to agree, and mixing the channels down
    /// is both cheaper and less prone to a transient in one channel being missed
    /// because the other was steady.
    fn detect(&mut self, samples: &[Vec<f32>]) -> Transient {
        let n = samples[0].len();
        let mut mix = vec![0.0f32; n];
        let scale = 1.0 / samples.len() as f32;
        for channel in samples {
            for (m, &s) in mix.iter_mut().zip(channel.iter()) {
                *m += s * scale;
            }
        }
        self.block_switch.analyse(&mix)
    }

    /// Encode the frame the lookahead has now made decidable.
    fn encode_pending(
        &mut self,
        samples: &[Vec<f32>],
        here: Transient,
        next: Transient,
    ) -> Result<Vec<u8>> {
        let num_ch = self.channels.len();
        self.decision = self.block_switch.decide(here, next);
        self.build_coding_table();
        self.transform(samples);

        // Noise shaping runs before anything measures the spectrum, because it is
        // the shaped residual that gets quantized and that the model has to judge.
        // It is a long-window tool here: an eight-short frame is already short
        // enough in time that there is little for it to do.
        let rate = self.config.sampling_rate;
        let bands = self.num_bands;
        let short = self.decision.sequence == WindowSequence::EightShortSequence;
        for ch in self.channels.iter_mut() {
            ch.tns = if short {
                None
            } else {
                apply_tns(
                    &mut ch.spectrum,
                    &self.sfb_offsets[..=bands],
                    bands.min(self.max_sfb),
                    rate,
                    false,
                )
            };
        }

        self.group_spectra();
        self.ms_used = false;
        if num_ch == 2 {
            self.decide_mid_side();
        }
        for c in 0..num_ch {
            self.analyse_channel(c);
        }
        self.allocate_and_fit(num_ch);

        self.write_frame(num_ch)
    }

    /// Window and transform one frame of every channel.
    fn transform(&mut self, samples: &[Vec<f32>]) {
        let n = self.config.frame_length.samples();
        let sequence = self.decision.sequence;
        let short_n = n / SUB_BLOCKS;

        // Both the shape a sequence uses and the previous frame's shape are fixed
        // here: the encoder emits sine windows throughout, so every join matches.
        let long = frame_window(n, sequence, WindowShape::Sine, WindowShape::Sine);
        let shorts: Vec<Vec<f32>> = if sequence == WindowSequence::EightShortSequence {
            (0..SUB_BLOCKS)
                .map(|i| short_window(n, WindowShape::Sine, WindowShape::Sine, i))
                .collect()
        } else {
            Vec::new()
        };

        for (c, ch) in self.channels.iter_mut().enumerate() {
            let input = &samples[c];
            ch.windowed[..n].copy_from_slice(&ch.history);
            ch.windowed[n..].copy_from_slice(input);

            if sequence == WindowSequence::EightShortSequence {
                let mut block = vec![0.0f32; 2 * short_n];
                for w in 0..SUB_BLOCKS {
                    let at = short_window_offset(n, w);
                    for (b, (&x, &g)) in
                        block.iter_mut().zip(ch.windowed[at..at + 2 * short_n].iter().zip(shorts[w].iter()))
                    {
                        *b = x * g;
                    }
                    let lo = w * short_n;
                    self.mdct_short.forward(
                        &block,
                        &mut ch.spectrum[lo..lo + short_n],
                        &mut self.mdct_scratch,
                    );
                }
            } else {
                let mut block = vec![0.0f32; 2 * n];
                for (b, (&x, &g)) in block.iter_mut().zip(ch.windowed.iter().zip(long.iter())) {
                    *b = x * g;
                }
                self.mdct.forward(&block, &mut ch.spectrum, &mut self.mdct_scratch);
            }

            ch.history.copy_from_slice(input);
        }
    }

    /// Build the band table the masking model and the rate loop work from.
    ///
    /// For a long frame that is just the band table. For an eight-short frame it is
    /// the short table repeated once per window group, each band widened to cover
    /// the group's windows, which is exactly the order the bitstream carries and the
    /// granularity a scalefactor applies at.
    fn build_coding_table(&mut self) {
        if self.decision.sequence != WindowSequence::EightShortSequence {
            self.coding_bands = self.num_bands.min(MAX_BANDS);
            self.coding_offsets[..=self.coding_bands]
                .copy_from_slice(&self.sfb_offsets[..=self.coding_bands]);
            return;
        }

        let mut at = 0usize;
        let mut band = 0usize;
        self.coding_offsets[0] = 0;
        for g in 0..self.decision.group_count {
            let length = self.decision.groups[g];
            for sfb in 0..self.short_max_sfb {
                if band >= MAX_BANDS {
                    break;
                }
                let width = self.short_offsets[sfb + 1] - self.short_offsets[sfb];
                at += width * length;
                band += 1;
                self.coding_offsets[band] = at;
            }
        }
        self.coding_bands = band;
    }

    /// Rearrange each channel's spectrum into the order the bitstream carries.
    fn group_spectra(&mut self) {
        let n = self.config.frame_length.samples();
        if self.decision.sequence != WindowSequence::EightShortSequence {
            for ch in self.channels.iter_mut() {
                ch.grouped[..n].copy_from_slice(&ch.spectrum[..n]);
            }
            return;
        }

        let short_n = n / SUB_BLOCKS;
        for ch in self.channels.iter_mut() {
            ch.grouped.fill(0.0);
            let mut at = 0usize;
            let mut window = 0usize;
            for g in 0..self.decision.group_count {
                let length = self.decision.groups[g];
                for sfb in 0..self.short_max_sfb {
                    let lo = self.short_offsets[sfb];
                    let hi = self.short_offsets[sfb + 1];
                    let width = hi - lo;
                    for w in 0..length {
                        let src = (window + w) * short_n + lo;
                        ch.grouped[at..at + width].copy_from_slice(&ch.spectrum[src..src + width]);
                        at += width;
                    }
                }
                window += length;
            }
        }
    }

    /// Bits the frame's payload may use, after the headers it has to carry.
    fn payload_budget(&self, num_ch: usize) -> usize {
        // ADTS header, element identifier and tag, the shared `ics_info`, the
        // mid/side mask, one `global_gain` per channel, and the terminator.
        let short = self.decision.sequence == WindowSequence::EightShortSequence;
        let ics_info = if short { 1 + 2 + 1 + 4 + 7 } else { 1 + 2 + 1 + 6 + 1 };
        let mask = if self.ms_used { self.coding_bands } else { 0 };
        let element = if num_ch >= 2 { 3 + 4 + 1 + ics_info + 2 + mask } else { 0 };
        let per_channel = if num_ch >= 2 { 8 } else { 3 + 4 + 8 + ics_info };
        let overhead = 56 + 3 + element + per_channel * num_ch;
        self.frame_bits.saturating_sub(overhead)
    }

    /// Split the frame's budget between channels and quantize each to its share.
    ///
    /// An even split wastes bits whenever the channels differ in difficulty, which
    /// after a mid/side decision they almost always do: a side channel carrying
    /// nothing would keep half the frame to itself. Shares therefore follow each
    /// channel's perceptual entropy, and whatever the first pass leaves unspent is
    /// handed back to the channel that can still use it.
    fn allocate_and_fit(&mut self, num_ch: usize) {
        let total = self.payload_budget(num_ch);
        let demand: Vec<f32> =
            (0..num_ch).map(|c| self.channels[c].psycho.perceptual_entropy.max(1.0)).collect();
        let sum: f32 = demand.iter().sum();

        let mut budgets: Vec<usize> =
            demand.iter().map(|d| (total as f32 * d / sum) as usize).collect();
        self.fit_all(&budgets);

        // One redistribution round: give the slack to the hungriest channel, which
        // is the one whose noise floor the extra bits will lower the most.
        for _ in 0..2 {
            let used: usize = (0..num_ch).map(|c| self.channels[c].coded.bits).sum();
            if used + total / 32 >= total {
                break;
            }
            let Some(hungriest) = (0..num_ch)
                .filter(|&c| self.channels[c].coded.bits >= budgets[c].saturating_sub(8))
                .max_by(|&a, &b| demand[a].total_cmp(&demand[b]))
            else {
                break;
            };
            budgets[hungriest] += total - used;
            self.fit_channel(hungriest, budgets[hungriest]);
        }
    }

    /// Choose which bands of a stereo pair to code as mid and side.
    ///
    /// Where the two channels are similar, the side signal is small and costs far
    /// fewer bits than a second full channel; where they are not, mid/side spreads
    /// each channel's noise into the other and costs quality. The usual test
    /// compares the energy the two representations would have to code, band by band,
    /// and takes whichever is smaller.
    fn decide_mid_side(&mut self) {
        let bands = self.coding_bands;
        let (left, right) = self.channels.split_at_mut(1);
        let left = &mut left[0];
        let right = &mut right[0];

        for b in 0..bands {
            let lo = self.coding_offsets[b];
            let hi = self.coding_offsets[b + 1];

            let mut lr = 0.0f64;
            let mut ms = 0.0f64;
            for i in lo..hi {
                let l = left.grouped[i] as f64;
                let r = right.grouped[i] as f64;
                lr += l * l + r * r;
                let m = 0.5 * (l + r);
                let s = 0.5 * (l - r);
                ms += m * m + s * s;
            }
            // A tie goes to left/right, which never makes anything worse.
            self.ms_mask[b] = ms < lr * 0.98;
        }

        // The mask is transmitted per band, so a frame with no band worth switching
        // saves the mask itself by declaring none.
        self.ms_used = self.ms_mask[..bands].iter().any(|&v| v);
        if !self.ms_used {
            return;
        }

        for b in 0..bands {
            if !self.ms_mask[b] {
                continue;
            }
            for i in self.coding_offsets[b]..self.coding_offsets[b + 1] {
                let l = left.grouped[i];
                let r = right.grouped[i];
                left.grouped[i] = 0.5 * (l + r);
                right.grouped[i] = 0.5 * (l - r);
            }
        }
    }

    /// Run the masking model for one channel.
    ///
    /// An eight-short frame is measured group by group, in time order, so that the
    /// model's pre-echo control works across the groups the way it does across
    /// frames — which is the whole point of splitting the frame up.
    fn analyse_channel(&mut self, c: usize) {
        let sequence = self.decision.sequence;
        let ch = &mut self.channels[c];

        if sequence != WindowSequence::EightShortSequence {
            let offsets = &self.coding_offsets[..=self.coding_bands];
            ch.model.analyse(&ch.grouped, offsets, sequence, &mut ch.psycho);
            return;
        }

        let mut group = PsychoResult::default();
        let mut energies = [0.0f32; MAX_BANDS];
        let mut band = 0usize;
        ch.psycho.bands = self.coding_bands;

        for g in 0..self.decision.group_count {
            let first = band;
            let count = self.short_max_sfb.min(self.coding_bands - first);
            for sfb in 0..count {
                let lo = self.coding_offsets[first + sfb];
                let hi = self.coding_offsets[first + sfb + 1];
                energies[sfb] = ch.grouped[lo..hi].iter().map(|&v| v * v).sum();
            }
            ch.short_model.analyse_energies(&energies[..count], sequence, &mut group);

            ch.psycho.energy[first..first + count].copy_from_slice(&group.energy[..count]);
            ch.psycho.threshold[first..first + count].copy_from_slice(&group.threshold[..count]);
            ch.psycho.spread_energy[first..first + count]
                .copy_from_slice(&group.spread_energy[..count]);
            if g == 0 {
                ch.psycho.perceptual_entropy = 0.0;
            }
            ch.psycho.perceptual_entropy += group.perceptual_entropy;
            band += count;
        }
    }

    /// Quantize one channel to fit `budget` payload bits.
    fn fit_channel(&mut self, c: usize, budget: usize) {
        let offsets = &self.coding_offsets[..=self.coding_bands];
        let frame = self.frame_count;
        let bands = self.short_max_sfb;
        let short = self.decision.sequence == WindowSequence::EightShortSequence;
        fit_one(&mut self.channels[c], offsets, budget, c, frame, short, bands);
    }


    /// Quantize every channel to its share of the budget.
    ///
    /// The channels share nothing at this point — each has its own spectrum, model
    /// and rate loop — so on a multi-channel frame they run in parallel.
    fn fit_all(&mut self, budgets: &[usize]) {
        let offsets = &self.coding_offsets[..=self.coding_bands];
        let frame = self.frame_count;
        let bands = self.short_max_sfb;
        let short = self.decision.sequence == WindowSequence::EightShortSequence;

        #[cfg(feature = "rayon")]
        if self.channels.len() > 2 {
            use rayon::prelude::*;
            self.channels
                .par_iter_mut()
                .zip(budgets.par_iter())
                .enumerate()
                .for_each(|(c, (ch, &budget))| {
                    fit_one(ch, offsets, budget, c, frame, short, bands)
                });
            return;
        }

        for (c, (ch, &budget)) in self.channels.iter_mut().zip(budgets.iter()).enumerate() {
            fit_one(ch, offsets, budget, c, frame, short, bands);
        }
    }

    /// The window layout the bitstream has to describe.
    fn layout(&self) -> Layout {
        let short = self.decision.sequence == WindowSequence::EightShortSequence;
        Layout {
            sequence: self.decision.sequence,
            max_sfb: if short { self.short_max_sfb } else { self.max_sfb },
            group_count: self.decision.group_count,
            grouping_bits: self.decision.grouping_bits(),
            num_swb: if short { self.short_bands } else { self.num_bands },
        }
    }

    /// Serialize the frame.
    fn write_frame(&mut self, num_ch: usize) -> Result<Vec<u8>> {
        let layout = self.layout();
        let offsets = self.coding_offsets;
        self.writer.reset();
        let w = &mut self.writer;

        match num_ch {
            1 => {
                w.write_u8(0, 3); // SCE
                w.write_u8(0, 4); // element instance tag
                write_channel(w, &self.channels[0], &offsets, &layout);
            }
            _ => {
                // Channels beyond the first pair are emitted as extra single
                // channel elements, which is legal for any channel count.
                w.write_u8(1, 3); // CPE
                w.write_u8(0, 4);
                w.write_bit(true); // common_window
                write_ics_info(w, &layout);
                if self.ms_used {
                    w.write_u8(1, 2); // ms_mask_present: per band
                    for b in 0..layout.coded_bands() {
                        w.write_bit(self.ms_mask[b]);
                    }
                } else {
                    w.write_u8(0, 2); // ms_mask_present: none
                }
                write_channel_body(w, &self.channels[0], &offsets, &layout);
                write_channel_body(w, &self.channels[1], &offsets, &layout);

                for ch in &self.channels[2..] {
                    w.write_u8(0, 3);
                    w.write_u8(0, 4);
                    write_channel(w, ch, &offsets, &layout);
                }
            }
        }

        w.write_u8(7, 3); // END
        w.byte_align_zero();
        let payload = w.as_bytes().to_vec();

        let mut frame = match self.config.output_format {
            OutputFormat::Adts => {
                // Prepend the ADTS header now that the payload length is known.
                let header = AdtsHeader {
                    mpeg_id: 0,
                    layer: 0,
                    protection_absent: true,
                    audio_object_type: self.config.audio_object_type,
                    sampling_rate: self.config.sampling_rate,
                    channel_config: self.config.channel_config,
                    frame_length: payload.len() + 7,
                    buffer_fullness: 0x7FF,
                    num_raw_data_blocks: 0,
                    crc: None,
                };
                let mut head = BitWriter::with_capacity(8);
                header.write(&mut head);
                head.byte_align_zero();
                head.into_bytes()
            }
            OutputFormat::Adif => {
                // One header before the first frame only; every frame after
                // that is the bare raw_data_block() with no framing of its
                // own, exactly what Decoder::decode_frame expects to follow
                // an ADIF header (see OutputFormat::Adif's docs).
                if self.adif_header_written {
                    Vec::new()
                } else {
                    self.adif_header_written = true;
                    let asc = AudioSpecificConfig {
                        audio_object_type: self.config.audio_object_type,
                        sampling_rate: self.config.sampling_rate,
                        channel_config: self.config.channel_config,
                        frame_length: self.config.frame_length,
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
                        original_copy: false,
                        home: false,
                        bitstream_type: true, // variable rate: no buffer_fullness field
                        bitrate: self.config.bitrate_bps,
                        num_program_config_elements: 1,
                        buffer_fullness: 0,
                        configs: vec![asc],
                    };
                    // The header's own bit length is not generally a multiple
                    // of 8, so the first frame's payload bits have to be
                    // packed on immediately after it in the same bitstream --
                    // byte-aligning the header first and concatenating bytes
                    // would insert padding a real ADIF stream (and this
                    // crate's own decoder, which does not skip any) does not
                    // expect. Every frame after this one already starts and
                    // ends on a byte boundary on its own (this encoder always
                    // byte-aligns a raw_data_block's payload), so only this
                    // one join has to happen bit by bit rather than
                    // byte by byte.
                    let mut head = BitWriter::with_capacity(payload.len() + 16);
                    header.write(&mut head);
                    let mut payload_bits = BitReader::new(&payload);
                    for _ in 0..payload.len() * 8 {
                        head.write_bit(payload_bits.read_bit().expect("just-written payload"));
                    }
                    head.byte_align_zero();
                    self.frame_count += 1;
                    return Ok(head.into_bytes());
                }
            }
        };
        frame.extend_from_slice(&payload);
        self.frame_count += 1;
        Ok(frame)
    }
}

/// Run one channel's rate loop.
///
/// Free rather than a method so that a parallel run borrows one channel at a time
/// instead of the whole encoder.
#[allow(clippy::too_many_arguments)]
fn fit_one(
    ch: &mut ChannelState,
    offsets: &[usize],
    budget: usize,
    index: usize,
    frame: u64,
    short: bool,
    short_bands: usize,
) {
    let ChannelState { grouped, psycho, coded, model, short_model, rate, tns, .. } = ch;
    // The short band table repeats once per window group, so the floor a band's
    // signal-to-mask ratio has repeats with it.
    let floor = |b: usize| {
        if short { short_model.min_snr(b % short_bands.max(1)) } else { model.min_snr(b) }
    };
    let bits = rate.fit(grouped, offsets, psycho, &floor, budget, coded);

    if std::env::var_os("AACENC_TRACE").is_some() {
        let peak_q = coded.quant.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
        let bands = coded.choices.iter().filter(|c| c.codebook != 0).count();
        let shaping = match tns {
            Some(f) => format!("order {} gain {:.2}", f.spec.order, f.gain),
            None => "off".to_string(),
        };
        eprintln!(
            "enc frame {} ch {index} bits {bits}/{budget} pe {:.0} coded_bands {bands} peak_q {peak_q} tns {shaping}",
            frame + 1,
            psycho.perceptual_entropy
        );
    }
}

/// Everything the bitstream needs to know about a frame's window layout.
#[derive(Debug, Clone, Copy)]
struct Layout {
    sequence: WindowSequence,
    /// Bands coded per window group.
    max_sfb: usize,
    /// Window groups in use; how many windows each holds is already folded into
    /// [`Self::coded_bands`] and does not need to be carried separately.
    group_count: usize,
    /// The `scale_factor_grouping` field.
    grouping_bits: u8,
    /// Bands the band table has, which the noise shaping field is counted from.
    num_swb: usize,
}

impl Layout {
    /// Bands the whole frame codes, across every group.
    #[inline]
    fn coded_bands(&self) -> usize {
        self.max_sfb * self.group_count
    }
}

/// Write `ics_info()`.
fn write_ics_info(w: &mut BitWriter, layout: &Layout) {
    w.write_bit(false); // ics_reserved_bit
    w.write_u8(layout.sequence as u8, 2);
    w.write_u8(0, 1); // sine window
    if layout.sequence == WindowSequence::EightShortSequence {
        w.write_u8(layout.max_sfb as u8, 4);
        w.write_u8(layout.grouping_bits, 7);
    } else {
        w.write_u8(layout.max_sfb as u8, 6);
        w.write_bit(false); // predictor_data_present
    }
}

/// Write a whole `individual_channel_stream()` including its `ics_info`.
fn write_channel(w: &mut BitWriter, ch: &ChannelState, offsets: &[usize], layout: &Layout) {
    w.write_u8(global_gain(ch, layout.coded_bands()), 8);
    write_ics_info(w, layout);
    write_ics_payload(w, ch, offsets, layout);
}

/// Write an `individual_channel_stream()` whose `ics_info` came from the element.
fn write_channel_body(w: &mut BitWriter, ch: &ChannelState, offsets: &[usize], layout: &Layout) {
    w.write_u8(global_gain(ch, layout.coded_bands()), 8);
    write_ics_payload(w, ch, offsets, layout);
}

/// The `global_gain` field, which the scalefactor deltas are counted from.
///
/// The decoder starts its running scalefactor at this value and adds the first
/// coded band's delta to it like any other, so setting it to that band's
/// scalefactor makes the first delta zero.
fn global_gain(ch: &ChannelState, bands: usize) -> u8 {
    let first = ch.coded.first_coded.filter(|&b| b < bands);
    match first {
        Some(b) => ch.coded.scalefactors[b].clamp(0, 255) as u8,
        None => SF_OFFSET as u8,
    }
}

/// Write `tns_data()` for a long window carrying one filter.
///
/// Band ranges travel as a length counted down from the total band count, which the
/// decoder then clips to the highest band the standard lets TNS reach. Since the
/// filter was applied over exactly that clipped range, transmitting the whole
/// span below its start says the same thing in fewer fields than a second,
/// order-zero filter would.
fn write_tns_data(w: &mut BitWriter, filter: &TnsFilter, num_swb: usize) {
    let spec = &filter.spec;
    let start = spec.start_band.min(num_swb);

    w.write_u8(1, 2); // n_filt
    w.write_u8(spec.resolution, 1);
    w.write_u8((num_swb - start) as u8, 6); // length, counted down from num_swb
    w.write_u8(spec.order as u8, 5);
    if spec.order > 0 {
        w.write_bit(spec.downward);
        w.write_bit(false); // coef_compress
        let width = spec.resolution as usize + 3;
        let mask = (1u32 << width) - 1;
        for i in 0..spec.order {
            w.write_u32(spec.coef[i] as u32 & mask, width);
        }
    }
}

/// Write section data, scalefactors, tool flags and spectral data.
fn write_ics_payload(w: &mut BitWriter, ch: &ChannelState, offsets: &[usize], layout: &Layout) {
    // Section data: run-length runs of equal codebooks, restarted at each group
    // because a run may not cross a group boundary. The length field is narrower
    // for short windows, where a group spans fewer bands.
    let (length_bits, escape) =
        if layout.sequence == WindowSequence::EightShortSequence { (3usize, 7usize) } else { (5, 31) };

    for g in 0..layout.group_count {
        let base = g * layout.max_sfb;
        let mut b = 0usize;
        while b < layout.max_sfb {
            let cb = ch.coded.choices[base + b].codebook;
            let mut run = 1usize;
            while b + run < layout.max_sfb && ch.coded.choices[base + b + run].codebook == cb {
                run += 1;
            }
            w.write_u8(cb, 4);
            let mut left = run;
            while left >= escape {
                w.write_u8(escape as u8, length_bits);
                left -= escape;
            }
            w.write_u8(left as u8, length_bits);
            b += run;
        }
    }

    // Scalefactor data: a DPCM delta for every band whose codebook is not ZERO,
    // the first one counted from `global_gain`.
    let mut previous: Option<i32> = None;
    for b in 0..layout.coded_bands() {
        if ch.coded.choices[b].codebook == 0 {
            continue;
        }
        let sf = ch.coded.scalefactors[b];
        let delta = match previous {
            Some(p) => sf - p,
            None => 0,
        };
        previous = Some(sf);
        write_scalefactor_delta(w, delta);
    }

    w.write_bit(false); // pulse_data_present
    match &ch.tns {
        Some(filter) => {
            w.write_bit(true);
            write_tns_data(w, filter, layout.num_swb);
        }
        None => w.write_bit(false),
    }
    w.write_bit(false); // gain_control_data_present

    for b in 0..layout.coded_bands() {
        let cb = ch.coded.choices[b].codebook;
        if cb == 0 {
            continue;
        }
        let lo = offsets[b];
        let hi = offsets[b + 1];
        write_band(w, cb, &ch.coded.quant[lo..hi]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(channels: usize, n: usize, freq: f32, rate: f32, phase: usize) -> AudioBuffer<i16> {
        let mut buf = AudioBuffer::<i16>::new(channels, n);
        for c in 0..channels {
            let data = buf.channel_mut(c);
            for (i, s) in data.iter_mut().enumerate() {
                let t = (phase + i) as f32 / rate;
                *s = ((t * freq * std::f32::consts::TAU).sin() * 12000.0) as i16;
            }
        }
        buf
    }

    /// A frame must carry real payload, not just headers.
    #[test]
    fn frames_carry_spectral_data() {
        let mut enc = Encoder::new(EncoderConfig::default()).unwrap();
        let mut sizes = Vec::new();
        for f in 0..8 {
            let pcm = tone(2, 1024, 440.0, 44100.0, f * 1024);
            sizes.push(enc.encode_frame(&pcm).unwrap().len());
        }
        // The first frame windows against silence, so judge from the second on.
        let steady = &sizes[1..];
        let min = *steady.iter().min().unwrap();
        assert!(min > 100, "frames are nearly empty: {sizes:?}");
    }

    /// Frame sizes must track the requested bitrate.
    #[test]
    fn frame_size_tracks_bitrate() {
        let mut measured = Vec::new();
        for bitrate in [64_000u32, 128_000, 256_000] {
            let config = EncoderConfig { bitrate_bps: bitrate, ..Default::default() };
            let mut enc = Encoder::new(config).unwrap();
            let mut total = 0usize;
            for f in 0..20 {
                let pcm = tone(2, 1024, 440.0, 44100.0, f * 1024);
                total += enc.encode_frame(&pcm).unwrap().len();
            }
            measured.push((bitrate, total));
        }
        for pair in measured.windows(2) {
            assert!(
                pair[1].1 > pair[0].1,
                "{} bps produced {} bytes, {} bps produced {}",
                pair[0].0,
                pair[0].1,
                pair[1].0,
                pair[1].1
            );
        }
    }

    /// Every emitted frame must be a well-formed ADTS frame whose declared length
    /// matches what was produced.
    #[test]
    fn frames_are_well_formed_adts() {
        use crate::bitstream::BitReader;
        let mut enc = Encoder::new(EncoderConfig::default()).unwrap();
        let mut frames = Vec::new();
        for f in 0..8 {
            let pcm = tone(2, 1024, 1000.0, 44100.0, f * 1024);
            frames.push(enc.encode_frame(&pcm).unwrap());
        }
        frames.push(enc.flush().unwrap());
        // The lookahead holds the first frame back, so it lands empty and every
        // real frame is delayed by one; that is fine, but each real frame still has
        // to be a well-formed ADTS frame.
        let real: Vec<_> = frames.into_iter().filter(|f| !f.is_empty()).collect();
        assert_eq!(real.len(), 8, "one held-back frame in, one flushed out");
        for frame in &real {
            let mut r = BitReader::new(frame);
            let header = AdtsHeader::parse(&mut r).expect("header parses");
            assert_eq!(header.frame_length, frame.len(), "declared length mismatch");
            assert_eq!(header.sampling_rate, SamplingRate::Hz44100);
            assert_eq!(header.channel_config, ChannelConfiguration::Stereo);
        }
    }

    /// A real transient must switch to short windows, and the result must still be
    /// a decodable stream: this is the end-to-end check for the whole lookahead,
    /// grouping and short-window bitstream path, not just one piece of it.
    #[test]
    fn transient_triggers_short_windows_and_decodes() {
        use crate::decoder::engine::Decoder;

        let mut enc = Encoder::new(EncoderConfig::default()).unwrap();
        let mut dec = Decoder::new_default();
        let mut saw_short = false;

        let mut rng = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 40) as i16
        };

        for f in 0..40u32 {
            let mut pcm = AudioBuffer::<i16>::new(2, 1024);
            // A loud burst of noise in one frame among otherwise pure quiet is as
            // sharp an attack as a real signal offers.
            if f == 20 {
                for c in 0..2 {
                    for s in pcm.channel_mut(c) {
                        *s = next();
                    }
                }
            }
            let frame = enc.encode_frame(&pcm).unwrap();
            if WindowSequence::EightShortSequence == enc.decision.sequence {
                saw_short = true;
            }
            if !frame.is_empty() {
                dec.decode_frame(&frame).expect("transient frame must decode");
            }
        }
        let frame = enc.flush().unwrap();
        if !frame.is_empty() {
            dec.decode_frame(&frame).expect("flushed frame must decode");
        }

        assert!(saw_short, "a sharp attack must trigger eight-short windows");
    }

    /// Mono must work as well as stereo.
    #[test]
    fn mono_encodes() {
        let config = EncoderConfig {
            channel_config: ChannelConfiguration::Mono,
            ..Default::default()
        };
        let mut enc = Encoder::new(config).unwrap();
        let mut frames = Vec::new();
        for f in 0..4 {
            let pcm = tone(1, 1024, 440.0, 44100.0, f * 1024);
            frames.push(enc.encode_frame(&pcm).unwrap());
        }
        frames.push(enc.flush().unwrap());
        for frame in frames.iter().filter(|f| !f.is_empty()) {
            assert!(frame.len() > 20, "mono frame too small: {}", frame.len());
        }
    }

    /// `OutputFormat::Adif` must produce a real, decodable stream: exactly one
    /// header (joined to the first frame bit for bit, not byte-aligned into
    /// it), every later frame bare, and every chunk decodes back through this
    /// crate's own `Decoder` -- including reconfiguring itself from the
    /// header the way a real player encountering this stream cold would.
    #[test]
    fn adif_output_round_trips_through_the_real_decoder() {
        use crate::decoder::engine::Decoder;
        use crate::syntax::asc::AudioSpecificConfig;

        let config = EncoderConfig { output_format: OutputFormat::Adif, ..Default::default() };
        let mut enc = Encoder::new(config).unwrap();
        let mut chunks = Vec::new();
        for f in 0..4 {
            let pcm = tone(2, 1024, 440.0, 44100.0, f * 1024);
            chunks.push(enc.encode_frame(&pcm).unwrap());
        }
        chunks.push(enc.flush().unwrap());
        let chunks: Vec<_> = chunks.into_iter().filter(|c| !c.is_empty()).collect();
        assert!(chunks.len() >= 4, "expected several real frames, got {}", chunks.len());

        // The first chunk must start with the ADIF syncword and be larger
        // than a bare frame; later chunks must NOT carry another header.
        assert_eq!(u32::from_be_bytes(chunks[0][..4].try_into().unwrap()), AdifHeader::SYNCWORD);
        for later in &chunks[1..] {
            assert_ne!(
                u32::from_be_bytes(later[..4.min(later.len())].try_into().unwrap_or([0; 4])),
                AdifHeader::SYNCWORD,
                "only the first chunk may carry the ADIF header"
            );
        }

        // A cold decoder, deliberately mis-configured, must reconfigure
        // itself from the header on the very first chunk (mirroring
        // `an_adif_prefixed_stream_decodes_and_reconfigures`) and then decode
        // every remaining bare chunk without any special handling.
        let mut dec = Decoder::new(AudioSpecificConfig {
            audio_object_type: AudioObjectType::AacLc,
            sampling_rate: SamplingRate::Hz16000,
            channel_config: ChannelConfiguration::Mono,
            frame_length: FrameLength::Samples1024,
            depends_on_core_coder: false,
            core_coder_delay: 0,
            extension_audio_object_type: None,
            extension_sampling_rate: None,
            sbr_present: false,
            ps_present: false,
        });
        for chunk in &chunks {
            dec.decode_frame(chunk).expect("every ADIF-framed chunk must decode");
        }
        assert_eq!(dec.channels(), 2, "must have reconfigured to stereo from the ADIF header");
        assert_eq!(dec.sample_rate_hz(), SamplingRate::Hz44100.hz());
    }

    /// Silence must still produce valid frames, and small ones.
    #[test]
    fn silence_encodes_compactly() {
        let mut enc = Encoder::new(EncoderConfig::default()).unwrap();
        let pcm = AudioBuffer::<i16>::new(2, 1024);
        let mut frames = Vec::new();
        for _ in 0..4 {
            frames.push(enc.encode_frame(&pcm).unwrap());
        }
        frames.push(enc.flush().unwrap());
        for frame in frames.iter().filter(|f| !f.is_empty()) {
            assert!(frame.len() >= 7, "frame shorter than its header");
            assert!(frame.len() < 200, "silence should compress hard, got {}", frame.len());
        }
    }
}
