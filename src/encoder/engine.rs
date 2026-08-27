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

use crate::bitstream::BitWriter;
use crate::buffer::AudioBuffer;
use crate::dsp::fft::Complex32;
use crate::dsp::mdct::MdctContext;
use crate::dsp::window::generate_sine_window_f32;
use crate::encoder::aac::huffman::write_scalefactor_delta;
use crate::encoder::aac::psycho::{PsychoResult, PsychoacousticModel};
use crate::encoder::aac::quant::{SF_OFFSET, write_band};
use crate::encoder::aac::rate::{Quantization, RateLoop};
use crate::encoder::aac::tns::{TnsFilter, apply as apply_tns};
use crate::error::Result;
use crate::syntax::adts::AdtsHeader;
use crate::tables::scalefactor::{MAX_SFB_LONG, compute_sfb_offsets, get_sfb_table};
use crate::types::{AudioObjectType, ChannelConfiguration, FrameLength, SamplingRate};

/// Encoder configuration.
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
            bitrate_bps: 128_000,
            frame_length: FrameLength::Samples1024,
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
    /// The masking model, which carries pre-echo state between frames.
    model: PsychoacousticModel,
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
    ) -> Self {
        Self {
            history: vec![0.0; n],
            windowed: vec![0.0; 2 * n],
            spectrum: vec![0.0; n],
            psycho: PsychoResult::default(),
            coded: Quantization::new(n, bands),
            model: PsychoacousticModel::new(sample_rate_hz, bitrate_per_channel, offsets, false),
            tns: None,
            rate: RateLoop::new(n),
        }
    }
}

/// AAC-LC encoder.
#[derive(Debug, Clone)]
pub struct Encoder {
    config: EncoderConfig,
    mdct: MdctContext,
    window: Vec<f32>,
    channels: Vec<ChannelState>,
    /// Cumulative band offsets, `num_bands + 1` entries.
    sfb_offsets: [usize; MAX_SFB_LONG + 1],
    num_bands: usize,
    /// Highest band the encoder codes.
    max_sfb: usize,
    /// Bit budget for one frame, from the requested bitrate.
    frame_bits: usize,
    frame_count: u64,
    mdct_scratch: Vec<Complex32>,
    writer: BitWriter,
    /// Bands coded mid/side, when the frame is a stereo pair.
    ms_mask: Vec<bool>,
    /// Whether any band of the current frame is coded mid/side.
    ms_used: bool,
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

        let mdct = MdctContext::new(n);
        let scratch_len = mdct.scratch_len();
        let config_rate = config.sampling_rate.hz();
        let per_channel_bitrate = config.bitrate_bps / num_ch as u32;

        Ok(Self {
            config,
            mdct,
            window: generate_sine_window_f32(2 * n),
            channels: (0..num_ch)
                .map(|_| {
                    ChannelState::new(
                        n,
                        num_bands,
                        config_rate,
                        per_channel_bitrate,
                        &sfb_offsets[..count],
                    )
                })
                .collect(),
            sfb_offsets,
            num_bands,
            max_sfb,
            frame_bits,
            frame_count: 0,
            mdct_scratch: vec![Complex32::default(); scratch_len],
            writer: BitWriter::with_capacity(4096),
            ms_mask: vec![false; num_bands],
            ms_used: false,
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
        }
        self.frame_count = 0;
    }

    /// Encode one frame of PCM into a complete ADTS frame.
    pub fn encode_frame(&mut self, pcm: &AudioBuffer<i16>) -> Result<Vec<u8>> {
        let num_ch = self.channels.len();
        let n = self.config.frame_length.samples();
        assert_eq!(pcm.channels(), num_ch, "input channel count mismatch");
        assert_eq!(pcm.samples_per_channel(), n, "input frame length mismatch");

        // Transform each channel.
        for c in 0..num_ch {
            let ch = &mut self.channels[c];
            let input = pcm.channel(c);

            for i in 0..n {
                ch.windowed[i] = ch.history[i] * self.window[i];
                ch.windowed[n + i] = input[i] as f32 * self.window[n + i];
            }
            self.mdct.forward(&ch.windowed, &mut ch.spectrum, &mut self.mdct_scratch);

            for (h, &s) in ch.history.iter_mut().zip(input.iter()) {
                *h = s as f32;
            }
        }

        // Noise shaping runs before anything measures the spectrum, because it is
        // the shaped residual that gets quantized and that the model has to judge.
        let rate = self.config.sampling_rate;
        let bands = self.num_bands;
        for ch in self.channels.iter_mut() {
            ch.tns = apply_tns(
                &mut ch.spectrum,
                &self.sfb_offsets[..=bands],
                bands.min(self.max_sfb),
                rate,
                false,
            );
        }

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

    /// Bits the frame's payload may use, after the headers it has to carry.
    fn payload_budget(&self, num_ch: usize) -> usize {
        // ADTS header, element identifier and tag, the shared `ics_info`, the
        // mid/side mask, one `global_gain` per channel, and the terminator.
        let element = if num_ch >= 2 { 3 + 4 + 1 + 11 + 2 + self.max_sfb } else { 0 };
        let per_channel = if num_ch >= 2 { 8 } else { 3 + 4 + 8 + 11 };
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
        let bands = self.num_bands.min(self.max_sfb);
        let (left, right) = self.channels.split_at_mut(1);
        let left = &mut left[0];
        let right = &mut right[0];

        for b in 0..bands {
            let lo = self.sfb_offsets[b];
            let hi = self.sfb_offsets[b + 1];

            let mut lr = 0.0f64;
            let mut ms = 0.0f64;
            for i in lo..hi {
                let l = left.spectrum[i] as f64;
                let r = right.spectrum[i] as f64;
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
            for i in self.sfb_offsets[b]..self.sfb_offsets[b + 1] {
                let l = left.spectrum[i];
                let r = right.spectrum[i];
                left.spectrum[i] = 0.5 * (l + r);
                right.spectrum[i] = 0.5 * (l - r);
            }
        }
    }

    /// Run the masking model for one channel.
    fn analyse_channel(&mut self, c: usize) {
        let offsets = &self.sfb_offsets[..=self.num_bands];
        let ch = &mut self.channels[c];
        ch.model.analyse(
            &ch.spectrum,
            offsets,
            crate::types::WindowSequence::OnlyLongSequence,
            &mut ch.psycho,
        );
    }

    /// Quantize one channel to fit `budget` payload bits.
    fn fit_channel(&mut self, c: usize, budget: usize) {
        let offsets = &self.sfb_offsets[..=self.num_bands];
        let frame = self.frame_count;
        fit_one(&mut self.channels[c], offsets, budget, c, frame);
    }


    /// Quantize every channel to its share of the budget.
    ///
    /// The channels share nothing at this point — each has its own spectrum, model
    /// and rate loop — so on a multi-channel frame they run in parallel.
    fn fit_all(&mut self, budgets: &[usize]) {
        let offsets = &self.sfb_offsets[..=self.num_bands];
        let frame = self.frame_count;

        #[cfg(feature = "rayon")]
        if self.channels.len() > 2 {
            use rayon::prelude::*;
            self.channels
                .par_iter_mut()
                .zip(budgets.par_iter())
                .enumerate()
                .for_each(|(c, (ch, &budget))| fit_one(ch, offsets, budget, c, frame));
            return;
        }

        for (c, (ch, &budget)) in self.channels.iter_mut().zip(budgets.iter()).enumerate() {
            fit_one(ch, offsets, budget, c, frame);
        }
    }

    /// Serialize the frame.
    fn write_frame(&mut self, num_ch: usize) -> Result<Vec<u8>> {
        self.writer.reset();
        let w = &mut self.writer;

        match num_ch {
            1 => {
                w.write_u8(0, 3); // SCE
                w.write_u8(0, 4); // element instance tag
                write_channel(w, &self.channels[0], &self.sfb_offsets, self.max_sfb, self.num_bands);
            }
            _ => {
                // Channels beyond the first pair are emitted as extra single
                // channel elements, which is legal for any channel count.
                w.write_u8(1, 3); // CPE
                w.write_u8(0, 4);
                w.write_bit(true); // common_window
                write_ics_info(w, self.max_sfb);
                if self.ms_used {
                    w.write_u8(1, 2); // ms_mask_present: per band
                    for b in 0..self.max_sfb {
                        w.write_bit(self.ms_mask[b]);
                    }
                } else {
                    w.write_u8(0, 2); // ms_mask_present: none
                }
                write_channel_body(w, &self.channels[0], &self.sfb_offsets, self.max_sfb, self.num_bands);
                write_channel_body(w, &self.channels[1], &self.sfb_offsets, self.max_sfb, self.num_bands);

                for ch in &self.channels[2..] {
                    w.write_u8(0, 3);
                    w.write_u8(0, 4);
                    write_channel(w, ch, &self.sfb_offsets, self.max_sfb, self.num_bands);
                }
            }
        }

        w.write_u8(7, 3); // END
        w.byte_align_zero();
        let payload = w.as_bytes().to_vec();

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

        let mut frame = head.into_bytes();
        frame.extend_from_slice(&payload);
        self.frame_count += 1;
        Ok(frame)
    }
}

/// Run one channel's rate loop.
///
/// Free rather than a method so that a parallel run borrows one channel at a time
/// instead of the whole encoder.
fn fit_one(ch: &mut ChannelState, offsets: &[usize], budget: usize, index: usize, frame: u64) {
    let ChannelState { spectrum, psycho, coded, model, rate, tns, .. } = ch;
    let bits = rate.fit(spectrum, offsets, psycho, &|b| model.min_snr(b), budget, coded);

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

/// Write `ics_info()` for a long window.
fn write_ics_info(w: &mut BitWriter, max_sfb: usize) {
    w.write_bit(false); // ics_reserved_bit
    w.write_u8(0, 2); // ONLY_LONG_SEQUENCE
    w.write_u8(0, 1); // sine window
    w.write_u8(max_sfb as u8, 6);
    w.write_bit(false); // predictor_data_present
}

/// Write a whole `individual_channel_stream()` including its `ics_info`.
fn write_channel(
    w: &mut BitWriter,
    ch: &ChannelState,
    offsets: &[usize],
    max_sfb: usize,
    num_swb: usize,
) {
    w.write_u8(global_gain(ch, max_sfb), 8);
    write_ics_info(w, max_sfb);
    write_ics_payload(w, ch, offsets, max_sfb, num_swb);
}

/// Write an `individual_channel_stream()` whose `ics_info` came from the element.
fn write_channel_body(
    w: &mut BitWriter,
    ch: &ChannelState,
    offsets: &[usize],
    max_sfb: usize,
    num_swb: usize,
) {
    w.write_u8(global_gain(ch, max_sfb), 8);
    write_ics_payload(w, ch, offsets, max_sfb, num_swb);
}

/// The `global_gain` field, which the scalefactor deltas are counted from.
///
/// The decoder starts its running scalefactor at this value and adds the first
/// coded band's delta to it like any other, so setting it to that band's
/// scalefactor makes the first delta zero.
fn global_gain(ch: &ChannelState, max_sfb: usize) -> u8 {
    let first = ch.coded.first_coded.filter(|&b| b < max_sfb);
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
fn write_ics_payload(
    w: &mut BitWriter,
    ch: &ChannelState,
    offsets: &[usize],
    max_sfb: usize,
    num_swb: usize,
) {
    // Section data: run-length runs of equal codebooks, with escape coding for
    // runs longer than the 5-bit length field can hold.
    let mut b = 0usize;
    while b < max_sfb {
        let cb = ch.coded.choices[b].codebook;
        let mut run = 1usize;
        while b + run < max_sfb && ch.coded.choices[b + run].codebook == cb {
            run += 1;
        }
        w.write_u8(cb, 4);
        let mut left = run;
        while left >= 31 {
            w.write_u8(31, 5);
            left -= 31;
        }
        w.write_u8(left as u8, 5);
        b += run;
    }

    // Scalefactor data: a DPCM delta for every band whose codebook is not ZERO,
    // the first one counted from `global_gain`.
    let mut previous: Option<i32> = None;
    for b in 0..max_sfb {
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
            write_tns_data(w, filter, num_swb);
        }
        None => w.write_bit(false),
    }
    w.write_bit(false); // gain_control_data_present

    for b in 0..max_sfb {
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
        for f in 0..8 {
            let pcm = tone(2, 1024, 1000.0, 44100.0, f * 1024);
            let frame = enc.encode_frame(&pcm).unwrap();
            let mut r = BitReader::new(&frame);
            let header = AdtsHeader::parse(&mut r).expect("header parses");
            assert_eq!(header.frame_length, frame.len(), "declared length mismatch");
            assert_eq!(header.sampling_rate, SamplingRate::Hz44100);
            assert_eq!(header.channel_config, ChannelConfiguration::Stereo);
        }
    }

    /// Mono must work as well as stereo.
    #[test]
    fn mono_encodes() {
        let config = EncoderConfig {
            channel_config: ChannelConfiguration::Mono,
            ..Default::default()
        };
        let mut enc = Encoder::new(config).unwrap();
        for f in 0..4 {
            let pcm = tone(1, 1024, 440.0, 44100.0, f * 1024);
            let frame = enc.encode_frame(&pcm).unwrap();
            assert!(frame.len() > 20, "mono frame too small: {}", frame.len());
        }
    }

    /// Silence must still produce valid frames, and small ones.
    #[test]
    fn silence_encodes_compactly() {
        let mut enc = Encoder::new(EncoderConfig::default()).unwrap();
        let pcm = AudioBuffer::<i16>::new(2, 1024);
        for _ in 0..4 {
            let frame = enc.encode_frame(&pcm).unwrap();
            assert!(frame.len() >= 7, "frame shorter than its header");
            assert!(frame.len() < 200, "silence should compress hard, got {}", frame.len());
        }
    }
}
