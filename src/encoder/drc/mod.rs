//! Loudness measurement and dynamic range metadata.
//!
//! An encoder that wants players to be able to level-match its output has to say how
//! loud the programme is. This module measures that the way ITU-R BS.1770 defines it
//! — K-weighted, gated, summed across channels — and turns the result, together with
//! a compression curve, into the `dynamic_range_info()` element the decoder in
//! [`crate::decoder::drc`] reads back.
//!
//! # The measurement
//!
//! BS.1770 filters each channel with two biquads that together approximate the way
//! the head and the ear weight a signal, takes the mean square over 400 ms blocks
//! that overlap by three quarters, and averages those blocks in the power domain —
//! but only the ones that are loud enough to count. Two gates decide that: an
//! absolute one at -70 LKFS, which drops silence, and a relative one ten units below
//! the ungated average, which drops the quiet parts of a programme that has any.
//!
//! The filter coefficients are given in the standard only for 48 kHz; here they are
//! derived from the underlying analogue prototypes, so any sampling rate measures on
//! the same curve rather than an aliased version of it.

use crate::bitstream::BitWriter;
use crate::decoder::drc::{DrcInfo, MAX_DRC_BANDS};

/// Length of one measurement block, in milliseconds.
const BLOCK_MS: usize = 400;
/// How far each block overlaps the last, as a fraction.
const BLOCK_OVERLAP: usize = 4;
/// Blocks quieter than this never count, in LKFS.
const ABSOLUTE_GATE_LKFS: f32 = -70.0;
/// How far below the ungated mean the relative gate sits, in loudness units.
const RELATIVE_GATE_LU: f32 = 10.0;
/// Offset that puts the mean square of a full-scale sine at 0 LKFS.
const LOUDNESS_OFFSET: f32 = -0.691;
/// Quarter decibels one step of a transmitted gain stands for.
const STEP_DB: f32 = 0.25;
/// Largest gain the seven-bit field can carry, in quarter decibels.
const MAX_GAIN_STEPS: i32 = 127;
/// Level a stream declares when it has nothing better to say, in quarter decibels
/// below full scale.
const DEFAULT_REFERENCE_LEVEL: f32 = 27.0;

/// A biquad in direct form I, which is what BS.1770 specifies.
#[derive(Debug, Clone, Copy)]
struct Biquad {
    b: [f32; 3],
    a: [f32; 2],
    x: [f32; 2],
    y: [f32; 2],
}

impl Biquad {
    const fn new(b: [f32; 3], a: [f32; 2]) -> Self {
        Self { b, a, x: [0.0; 2], y: [0.0; 2] }
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let output = self.b[0] * input + self.b[1] * self.x[0] + self.b[2] * self.x[1]
            - self.a[0] * self.y[0]
            - self.a[1] * self.y[1];
        self.x[1] = self.x[0];
        self.x[0] = input;
        self.y[1] = self.y[0];
        self.y[0] = output;
        output
    }

    fn reset(&mut self) {
        self.x = [0.0; 2];
        self.y = [0.0; 2];
    }
}

/// The two-stage K-weighting filter, for one channel.
#[derive(Debug, Clone, Copy)]
struct KWeighting {
    shelf: Biquad,
    highpass: Biquad,
}

impl KWeighting {
    /// Derive the filter for a sampling rate.
    ///
    /// Both stages come from the analogue prototypes the standard's 48 kHz
    /// coefficients were themselves derived from, bilinear-transformed at the rate
    /// in hand.
    fn new(sample_rate_hz: u32) -> Self {
        let fs = sample_rate_hz as f64;

        // Stage one: a high shelf standing in for the head's diffraction.
        let f0 = 1681.974_450_955_533;
        let gain_db = 3.999_843_853_973_347;
        let q = 0.707_175_236_955_419_6;
        let k = (std::f64::consts::PI * f0 / fs).tan();
        let vh = 10f64.powf(gain_db / 20.0);
        let vb = vh.powf(0.499_666_774_154_541_6);
        let a0 = 1.0 + k / q + k * k;
        let shelf = Biquad::new(
            [
                ((vh + vb * k / q + k * k) / a0) as f32,
                (2.0 * (k * k - vh) / a0) as f32,
                ((vh - vb * k / q + k * k) / a0) as f32,
            ],
            [(2.0 * (k * k - 1.0) / a0) as f32, ((1.0 - k / q + k * k) / a0) as f32],
        );

        // Stage two: the RLB high pass, which discounts what the ear barely hears.
        let f0 = 38.135_470_876_024_44;
        let q = 0.500_327_037_323_877_3;
        let k = (std::f64::consts::PI * f0 / fs).tan();
        let denominator = 1.0 + k / q + k * k;
        let highpass = Biquad::new(
            [1.0, -2.0, 1.0],
            [
                (2.0 * (k * k - 1.0) / denominator) as f32,
                ((1.0 - k / q + k * k) / denominator) as f32,
            ],
        );

        Self { shelf, highpass }
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        self.highpass.process(self.shelf.process(input))
    }

    fn reset(&mut self) {
        self.shelf.reset();
        self.highpass.reset();
    }
}

/// Weight BS.1770 gives each channel of a standard layout.
///
/// The surround channels count for more because a listener localises them less
/// precisely and judges them louder than their level alone suggests. Low-frequency
/// effects channels are excluded outright.
fn channel_weight(channel: usize, channels: usize) -> f32 {
    match (channels, channel) {
        // 5.1 and above: centre, left, right, LFE, then surrounds.
        (6, 3) | (8, 3) => 0.0,
        (6, 4..=5) | (8, 4..=7) => 1.41,
        _ => 1.0,
    }
}

/// An ITU-R BS.1770 loudness meter.
///
/// Feed it whole frames; ask for the integrated loudness whenever the answer is
/// wanted. It keeps only the per-block mean squares, so a long programme costs
/// memory linear in its length in blocks rather than in samples.
#[derive(Debug, Clone)]
pub struct LoudnessMeter {
    filters: Vec<KWeighting>,
    weights: Vec<f32>,
    /// Samples in one quarter of a block, which is how far blocks are apart.
    step: usize,
    /// Weighted sum of squares of each of the last four quarter-blocks, newest last.
    quarters: [f64; BLOCK_OVERLAP],
    /// Samples already in the quarter being filled.
    filled: usize,
    /// Quarter-blocks seen, so the first three do not close a block early.
    seen: usize,
    /// Mean square of every completed block.
    blocks: Vec<f64>,
}

impl LoudnessMeter {
    /// Build a meter for a layout and rate.
    pub fn new(sample_rate_hz: u32, channels: usize) -> Self {
        let channels = channels.max(1);
        let block_len = (sample_rate_hz as usize * BLOCK_MS / 1000).max(1);
        Self {
            filters: vec![KWeighting::new(sample_rate_hz); channels],
            weights: (0..channels).map(|c| channel_weight(c, channels)).collect(),
            step: (block_len / BLOCK_OVERLAP).max(1),
            quarters: [0.0; BLOCK_OVERLAP],
            filled: 0,
            seen: 0,
            blocks: Vec::new(),
        }
    }

    /// Forget everything measured so far.
    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
        self.quarters = [0.0; BLOCK_OVERLAP];
        self.filled = 0;
        self.seen = 0;
        self.blocks.clear();
    }

    /// Blocks measured so far.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Feed one frame, as one slice per channel, all the same length.
    ///
    /// Blocks overlap, so a sample contributes to several of them; the meter keeps
    /// the tail it still needs rather than asking the caller to overlap its frames.
    pub fn push(&mut self, channels: &[&[f32]]) {
        if channels.is_empty() {
            return;
        }
        let len = channels[0].len();

        // Blocks overlap by three quarters, so rather than keep the samples around,
        // keep the four most recent quarter-block sums: a block's mean square is
        // just their total.
        for i in 0..len {
            let mut weighted = 0.0f64;
            for (c, channel) in channels.iter().enumerate() {
                if c >= self.filters.len() || i >= channel.len() {
                    continue;
                }
                let filtered = self.filters[c].process(channel[i]);
                weighted += (self.weights[c] * filtered * filtered) as f64;
            }
            self.quarters[BLOCK_OVERLAP - 1] += weighted;
            self.filled += 1;

            if self.filled == self.step {
                self.seen += 1;
                if self.seen >= BLOCK_OVERLAP {
                    let total: f64 = self.quarters.iter().sum();
                    self.blocks.push(total / (self.step * BLOCK_OVERLAP) as f64);
                }
                self.quarters.rotate_left(1);
                self.quarters[BLOCK_OVERLAP - 1] = 0.0;
                self.filled = 0;
            }
        }
    }

    /// Integrated loudness in LKFS, or `None` before any block has completed.
    pub fn integrated_lkfs(&self) -> Option<f32> {
        if self.blocks.is_empty() {
            return None;
        }
        let absolute: Vec<f64> = self
            .blocks
            .iter()
            .copied()
            .filter(|&power| level(power) > ABSOLUTE_GATE_LKFS)
            .collect();
        if absolute.is_empty() {
            return Some(ABSOLUTE_GATE_LKFS);
        }

        let ungated = absolute.iter().sum::<f64>() / absolute.len() as f64;
        let threshold = level(ungated) - RELATIVE_GATE_LU;
        let gated: Vec<f64> =
            absolute.into_iter().filter(|&power| level(power) > threshold).collect();
        if gated.is_empty() {
            return Some(level(ungated));
        }
        Some(level(gated.iter().sum::<f64>() / gated.len() as f64))
    }
}

/// Turn a mean square into a loudness in LKFS.
#[inline]
fn level(power: f64) -> f32 {
    if power <= 0.0 {
        return f32::NEG_INFINITY;
    }
    LOUDNESS_OFFSET + 10.0 * power.log10() as f32
}

/// How hard to compress, which is what distinguishes one DRC profile from another.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressionProfile {
    /// Level above which the programme is turned down, in LKFS.
    pub knee_lkfs: f32,
    /// How much of the excess above the knee to remove, as a fraction.
    pub ratio: f32,
    /// Largest attenuation to ask for, in dB.
    pub max_cut_db: f32,
    /// Largest boost to ask for, in dB.
    pub max_boost_db: f32,
}

impl Default for CompressionProfile {
    /// A moderate profile: nothing below -24 LKFS is touched, and half of anything
    /// above it is taken off.
    fn default() -> Self {
        Self { knee_lkfs: -24.0, ratio: 0.5, max_cut_db: 12.0, max_boost_db: 6.0 }
    }
}

/// Measures loudness and writes the metadata a decoder needs to act on it.
#[derive(Debug, Clone)]
pub struct DrcEncoder {
    meter: LoudnessMeter,
    profile: CompressionProfile,
    /// Loudness the programme is meant to be played back at.
    target_lkfs: f32,
    /// Short-term meter for the frame being encoded.
    frame_meter: LoudnessMeter,
}

impl DrcEncoder {
    /// Build an encoder targeting `target_lkfs`, with the default profile.
    pub fn new(sample_rate_hz: u32, channels: usize, target_lkfs: f32) -> Self {
        Self {
            meter: LoudnessMeter::new(sample_rate_hz, channels),
            profile: CompressionProfile::default(),
            target_lkfs,
            frame_meter: LoudnessMeter::new(sample_rate_hz, channels),
        }
    }

    /// Change how hard the metadata asks a decoder to compress.
    pub fn set_profile(&mut self, profile: CompressionProfile) {
        self.profile = profile;
    }

    /// Integrated loudness of everything fed so far.
    pub fn integrated_lkfs(&self) -> Option<f32> {
        self.meter.integrated_lkfs()
    }

    /// Reference level to declare, in quarter decibels below full scale.
    ///
    /// This is what a decoder normalises against, so it has to describe the
    /// programme as a whole rather than the frame in hand.
    pub fn reference_level(&self) -> u8 {
        match self.meter.integrated_lkfs() {
            Some(lkfs) if lkfs.is_finite() => (-lkfs * 4.0).clamp(0.0, 127.0) as u8,
            _ => DEFAULT_REFERENCE_LEVEL as u8,
        }
    }

    /// Measure one frame and produce the metadata that describes it.
    ///
    /// `channels` is one slice per channel, all the same length.
    pub fn analyse(&mut self, channels: &[&[f32]]) -> DrcInfo {
        self.meter.push(channels);

        self.frame_meter.reset();
        self.frame_meter.push(channels);
        let short_term = self.frame_meter.integrated_lkfs().unwrap_or_else(|| {
            // A frame shorter than a measurement block leaves the meter empty, so
            // fall back on the plain mean square, which is what the block would have
            // been had it closed.
            mean_square_lkfs(channels)
        });

        let gain_db = self.gain_for(short_term);
        let steps = (gain_db / STEP_DB).round() as i32;

        let mut info = DrcInfo {
            bands: 1,
            band_top: [usize::MAX; MAX_DRC_BANDS],
            gain: [0; MAX_DRC_BANDS],
            reference_level: Some(self.reference_level()),
            channel_included: [true; 8],
            interpolation_scheme: 0,
        };
        info.gain[0] = steps.clamp(-MAX_GAIN_STEPS, MAX_GAIN_STEPS) as i8;
        info
    }

    /// The gain the profile asks for at a measured loudness.
    fn gain_for(&self, lkfs: f32) -> f32 {
        if !lkfs.is_finite() {
            return 0.0;
        }
        // Bring the programme to its target, then take a share of whatever still
        // stands above the knee.
        let normalisation = self.target_lkfs - lkfs;
        let excess = (lkfs - self.profile.knee_lkfs).max(0.0);
        let compression = -excess * self.profile.ratio;
        (normalisation.min(0.0) + compression)
            .clamp(-self.profile.max_cut_db, self.profile.max_boost_db)
    }
}

/// Loudness of a frame too short for a measurement block, from its mean square.
fn mean_square_lkfs(channels: &[&[f32]]) -> f32 {
    let mut power = 0.0f64;
    let mut count = 0usize;
    for channel in channels {
        for &sample in *channel {
            power += (sample as f64) * sample as f64;
            count += 1;
        }
    }
    if count == 0 { f32::NEG_INFINITY } else { level(power / count as f64) }
}

/// Write one `dynamic_range_info()` element.
///
/// The layout is the one [`DrcInfo::parse`] reads; a payload written here and parsed
/// there comes back unchanged.
pub fn write_dynamic_range_info(w: &mut BitWriter, info: &DrcInfo) {
    w.write_bit(false); // pce_tag_present
    let excludes_any = info.channel_included.iter().any(|&on| !on);
    w.write_bit(excludes_any);
    if excludes_any {
        for channel in 0..7 {
            w.write_bit(!info.channel_included[channel]);
        }
        w.write_bit(false); // no further channels
    }

    let multiband = info.bands > 1;
    w.write_bit(multiband);
    if multiband {
        w.write_u8(info.bands as u8 - 1, 4);
        w.write_u8(info.interpolation_scheme, 4);
        for b in 0..info.bands {
            w.write_u8((info.band_top[b] / 4 - 1) as u8, 8);
        }
    }

    match info.reference_level {
        Some(level) => {
            w.write_bit(true);
            w.write_u8(level, 7);
            w.write_bit(false);
        }
        None => w.write_bit(false),
    }

    for b in 0..info.bands {
        w.write_bit(info.gain[b] < 0);
        w.write_u8(info.gain[b].unsigned_abs(), 7);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitReader;

    fn sine(rate: u32, seconds: f32, freq: f32, amplitude: f32) -> Vec<f32> {
        let n = (rate as f32 * seconds) as usize;
        (0..n)
            .map(|i| amplitude * (std::f32::consts::TAU * freq * i as f32 / rate as f32).sin())
            .collect()
    }

    /// A 1 kHz sine at -20 dBFS measures -20 LKFS, which is the calibration the
    /// standard's own test signal defines.
    #[test]
    fn the_meter_matches_the_reference_signal() {
        let rate = 48000;
        let amplitude = 10f32.powf(-20.0 / 20.0) * std::f32::consts::SQRT_2;
        let signal = sine(rate, 5.0, 1000.0, amplitude);

        let mut meter = LoudnessMeter::new(rate, 1);
        meter.push(&[&signal]);
        let lkfs = meter.integrated_lkfs().expect("five seconds is many blocks");
        assert!((lkfs + 20.0).abs() < 0.15, "measured {lkfs} LKFS, expected -20");
    }

    /// The measurement must not depend on the sampling rate.
    #[test]
    fn the_filter_is_derived_for_any_rate() {
        let amplitude = 10f32.powf(-20.0 / 20.0) * std::f32::consts::SQRT_2;
        for rate in [32000u32, 44100, 48000, 96000] {
            let signal = sine(rate, 4.0, 1000.0, amplitude);
            let mut meter = LoudnessMeter::new(rate, 1);
            meter.push(&[&signal]);
            let lkfs = meter.integrated_lkfs().unwrap();
            assert!((lkfs + 20.0).abs() < 0.3, "{rate} Hz measured {lkfs} LKFS");
        }
    }

    /// Silence must fall below the absolute gate rather than report a level.
    #[test]
    fn silence_is_gated_out() {
        let mut meter = LoudnessMeter::new(48000, 2);
        let quiet = vec![0.0f32; 48000];
        meter.push(&[&quiet, &quiet]);
        assert_eq!(meter.integrated_lkfs(), Some(ABSOLUTE_GATE_LKFS));
    }

    /// The surround channels must count for more than the front ones.
    #[test]
    fn surround_channels_are_weighted_up() {
        let rate = 48000;
        let amplitude = 0.5;
        let tone = sine(rate, 3.0, 1000.0, amplitude);
        let quiet = vec![0.0f32; tone.len()];

        let mut front = LoudnessMeter::new(rate, 6);
        front.push(&[&tone, &quiet, &quiet, &quiet, &quiet, &quiet]);
        let mut surround = LoudnessMeter::new(rate, 6);
        surround.push(&[&quiet, &quiet, &quiet, &quiet, &tone, &quiet]);

        let front = front.integrated_lkfs().unwrap();
        let surround = surround.integrated_lkfs().unwrap();
        assert!(surround > front + 1.0, "front {front}, surround {surround}");
    }

    /// The low-frequency effects channel must not count at all.
    #[test]
    fn the_lfe_channel_is_excluded() {
        let rate = 48000;
        let tone = sine(rate, 3.0, 1000.0, 0.5);
        let quiet = vec![0.0f32; tone.len()];
        let mut meter = LoudnessMeter::new(rate, 6);
        meter.push(&[&quiet, &quiet, &quiet, &tone, &quiet, &quiet]);
        assert_eq!(meter.integrated_lkfs(), Some(ABSOLUTE_GATE_LKFS));
    }

    /// Metadata written here must parse back to the same thing.
    #[test]
    fn metadata_round_trips_through_the_decoder() {
        let rate = 48000;
        let loud = sine(rate, 1.0, 1000.0, 0.9);
        let mut encoder = DrcEncoder::new(rate, 1, -23.0);
        let info = encoder.analyse(&[&loud]);
        assert!(info.gain[0] < 0, "a loud programme should ask to be turned down");

        let mut w = BitWriter::with_capacity(32);
        write_dynamic_range_info(&mut w, &info);
        w.byte_align_zero();
        let bytes = w.into_bytes();

        let mut reader = BitReader::new(&bytes);
        let parsed = DrcInfo::parse(&mut reader).unwrap();
        assert_eq!(parsed.bands, info.bands);
        assert_eq!(parsed.gain[0], info.gain[0]);
        assert_eq!(parsed.reference_level, info.reference_level);
    }

    /// A quiet programme must not be asked to compress.
    #[test]
    fn a_quiet_programme_is_left_alone() {
        let rate = 48000;
        let quiet = sine(rate, 1.0, 1000.0, 0.01);
        let mut encoder = DrcEncoder::new(rate, 1, -23.0);
        let info = encoder.analyse(&[&quiet]);
        assert!(info.gain[0] >= 0, "a quiet programme was asked to duck");
    }
}
