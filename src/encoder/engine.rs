//! AAC-LC encoder.
//!
//! Each frame windows the current input together with the previous frame's samples,
//! transforms them with the MDCT, quantizes the result under a bit budget, and
//! writes an ADTS frame.
//!
//! # Scope
//!
//! This encoder emits long windows with a sine shape and a flat quantization noise
//! floor set by rate control alone. It produces conformant, decodable AAC-LC, but
//! it does not yet use the tools that buy perceptual quality at a given bitrate:
//! block switching for transients, mid/side stereo, TNS, or a masking model driving
//! per-band scalefactors. Those are the difference between "correct" and
//! "competitive", and they are not implemented here.

use crate::bitstream::BitWriter;
use crate::buffer::AudioBuffer;
use crate::dsp::fft::Complex32;
use crate::dsp::mdct::MdctContext;
use crate::dsp::window::generate_sine_window_f32;
use crate::encoder::aac::huffman::write_scalefactor_delta;
use crate::encoder::aac::quant::{
    BandChoice, SF_OFFSET, choose_codebook, quantize_band, write_band,
};
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
    /// Quantized coefficients.
    quant: Vec<i32>,
    /// Chosen codebook per band.
    choices: Vec<BandChoice>,
    /// Scalefactor per band.
    scalefactors: Vec<i32>,
}

impl ChannelState {
    fn new(n: usize, bands: usize) -> Self {
        Self {
            history: vec![0.0; n],
            windowed: vec![0.0; 2 * n],
            spectrum: vec![0.0; n],
            quant: vec![0; n],
            choices: vec![BandChoice::default(); bands],
            scalefactors: vec![SF_OFFSET; bands],
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

        Ok(Self {
            config,
            mdct,
            window: generate_sine_window_f32(2 * n),
            channels: (0..num_ch).map(|_| ChannelState::new(n, num_bands)).collect(),
            sfb_offsets,
            num_bands,
            max_sfb,
            frame_bits,
            frame_count: 0,
            mdct_scratch: vec![Complex32::default(); scratch_len],
            writer: BitWriter::with_capacity(4096),
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

        // Reserve room for the element headers and the ADTS header itself.
        let overhead_bits = 56 + num_ch * 24 + 3;
        let budget = self.frame_bits.saturating_sub(overhead_bits) / num_ch.max(1);
        for c in 0..num_ch {
            self.fit_channel(c, budget);
        }

        self.write_frame(num_ch)
    }

    /// Choose a scalefactor for channel `c` that fits its share of the frame budget.
    ///
    /// Bisects on a single scalefactor applied to every band: raising it coarsens
    /// the quantization monotonically, so cost is monotone in it and bisection is
    /// well defined.
    fn fit_channel(&mut self, c: usize, budget: usize) {
        // Search the whole legal scalefactor range. Cost falls monotonically as the
        // scalefactor rises, so bisection finds the smallest one that fits, which is
        // the finest quantization the budget allows.
        let mut lo = 0i32;
        let mut hi = 255i32;
        let mut best = 255i32;
        let mut found = false;

        while lo <= hi {
            let mid = (lo + hi) / 2;
            if self.cost_at(c, mid) <= budget {
                best = mid;
                found = true;
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }

        if !found {
            // Even the coarsest quantization overspends; take it and let the frame
            // run over rather than emitting nothing.
            best = 255;
        }

        // Re-run at the chosen scalefactor so the quantized values and codebook
        // choices left behind are the ones that will be written.
        let bits = self.cost_at(c, best);
        if std::env::var_os("AACENC_TRACE").is_some() {
            let ch = &self.channels[c];
            let peak_spec = ch.spectrum.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let peak_q = ch.quant.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
            let clamped = ch.quant.iter().filter(|v| v.unsigned_abs() >= 8191).count();
            eprintln!(
                "enc frame {} ch {c} sf {best} bits {bits}/{budget} peak_spec {peak_spec:.0} peak_q {peak_q} clamped {clamped}",
                self.frame_count + 1
            );
        }
        let ch = &mut self.channels[c];
        ch.scalefactors[..self.num_bands].fill(best);
    }

    /// Quantize channel `c` at `scalefactor` and return the payload bit cost.
    ///
    /// Leaves the quantized values and codebook choices in place, so the caller can
    /// re-run it at the chosen scalefactor before writing.
    fn cost_at(&mut self, c: usize, scalefactor: i32) -> usize {
        let max_sfb = self.max_sfb;
        let ch = &mut self.channels[c];

        let mut bits = 0usize;
        for b in 0..max_sfb {
            let lo = self.sfb_offsets[b];
            let hi = self.sfb_offsets[b + 1];
            quantize_band(&ch.spectrum[lo..hi], scalefactor, &mut ch.quant[lo..hi]);
            let choice = choose_codebook(&ch.quant[lo..hi]);
            ch.choices[b] = choice;
            bits += choice.bits as usize;
        }

        // Section data: one run-length record per codebook change.
        let mut sections = 1usize;
        for b in 1..max_sfb {
            if ch.choices[b].codebook != ch.choices[b - 1].codebook {
                sections += 1;
            }
        }
        bits += sections * (4 + 5);
        // Scalefactors: one Huffman-coded delta per coded band. All bands share one
        // scalefactor here, so every delta is zero; use that codeword's real length
        // rather than a guess, so the budget the rate loop enforces is exact.
        let zero_delta_bits = crate::encoder::aac::huffman::scalefactor_codeword(0)
            .map_or(2, |c| c.len as usize);
        let coded = ch.choices[..max_sfb].iter().filter(|c| c.codebook != 0).count();
        bits += coded * zero_delta_bits;
        bits
    }

    /// Serialize the frame.
    fn write_frame(&mut self, num_ch: usize) -> Result<Vec<u8>> {
        self.writer.reset();
        let w = &mut self.writer;

        match num_ch {
            1 => {
                w.write_u8(0, 3); // SCE
                w.write_u8(0, 4); // element instance tag
                write_channel(w, &self.channels[0], &self.sfb_offsets, self.max_sfb);
            }
            _ => {
                // Channels beyond the first pair are emitted as extra single
                // channel elements, which is legal for any channel count.
                w.write_u8(1, 3); // CPE
                w.write_u8(0, 4);
                w.write_bit(true); // common_window
                write_ics_info(w, self.max_sfb);
                w.write_u8(0, 2); // ms_mask_present = 0
                write_channel_body(w, &self.channels[0], &self.sfb_offsets, self.max_sfb);
                write_channel_body(w, &self.channels[1], &self.sfb_offsets, self.max_sfb);

                for ch in &self.channels[2..] {
                    w.write_u8(0, 3);
                    w.write_u8(0, 4);
                    write_channel(w, ch, &self.sfb_offsets, self.max_sfb);
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
) {
    w.write_u8((ch.scalefactors[0] & 0xFF) as u8, 8); // global_gain
    write_ics_info(w, max_sfb);
    write_ics_payload(w, ch, offsets, max_sfb);
}

/// Write an `individual_channel_stream()` whose `ics_info` came from the element.
fn write_channel_body(
    w: &mut BitWriter,
    ch: &ChannelState,
    offsets: &[usize],
    max_sfb: usize,
) {
    w.write_u8((ch.scalefactors[0] & 0xFF) as u8, 8); // global_gain
    write_ics_payload(w, ch, offsets, max_sfb);
}

/// Write section data, scalefactors, tool flags and spectral data.
fn write_ics_payload(
    w: &mut BitWriter,
    ch: &ChannelState,
    offsets: &[usize],
    max_sfb: usize,
) {
    // Section data: run-length runs of equal codebooks, with escape coding for
    // runs longer than the 5-bit length field can hold.
    let mut b = 0usize;
    while b < max_sfb {
        let cb = ch.choices[b].codebook;
        let mut run = 1usize;
        while b + run < max_sfb && ch.choices[b + run].codebook == cb {
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

    // Scalefactor data: the standard codes a DPCM delta for every band whose
    // codebook is not ZERO, including the first. Every band here shares the global
    // gain, so every delta is zero.
    for b in 0..max_sfb {
        if ch.choices[b].codebook != 0 {
            write_scalefactor_delta(w, 0);
        }
    }

    w.write_bit(false); // pulse_data_present
    w.write_bit(false); // tns_data_present
    w.write_bit(false); // gain_control_data_present

    for b in 0..max_sfb {
        let cb = ch.choices[b].codebook;
        if cb == 0 {
            continue;
        }
        let lo = offsets[b];
        let hi = offsets[b + 1];
        write_band(w, cb, &ch.quant[lo..hi]);
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
