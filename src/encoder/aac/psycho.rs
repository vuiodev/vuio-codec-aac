//! The masking model.
//!
//! A quantizer that spreads its noise evenly across the spectrum wastes bits: the
//! ear cannot hear noise sitting under a loud neighbouring tone, and hears very
//! little in a band that is quiet in absolute terms. This module works out, band by
//! band, how much noise the signal in and around that band will hide — the *masking
//! threshold* — so that the rate loop can spend bits where they are audible and
//! nowhere else.
//!
//! The chain, per frame and channel:
//!
//! 1. band energies, over the scalefactor bands the bitstream already uses;
//! 2. a threshold of `energy * 10^-2.9`, capped, which is the signal-to-mask ratio a
//!    band's own content buys;
//! 3. *spreading*: each band also masks its neighbours, less and less with Bark
//!    distance, and more downwards in frequency than upwards;
//! 4. the absolute threshold of hearing, which no signal has to beat;
//! 5. *pre-echo control*: a threshold may not rise more than a factor of two above
//!    the previous frame's, so that noise spread over a whole window by a sudden
//!    attack does not become audible in the quiet that preceded it.
//!
//! It also reports two things the rate loop needs: the *perceptual entropy*, an
//! estimate of the bits the frame needs to stay transparent, and a separately
//! spread copy of the energy used to spot bands that would otherwise be zeroed
//! into an audible hole.
//!
//! Everything here follows the reference encoder's model, which is the one the
//! constants are tuned for.

use crate::types::WindowSequence;

/// Bands the model can handle.
pub const MAX_BANDS: usize = 64;

/// Highest Bark value the tables run to.
const MAX_BARK: f32 = 24.0;
/// Spreading slope towards lower frequency, dB per Bark.
const MASK_LOW_FACTOR: f32 = 3.0;
/// Spreading slope towards higher frequency, dB per Bark.
const MASK_HIGH_FACTOR: f32 = 1.5;
/// Slopes for the separately spread energy used to spot holes, long windows.
const HOLE_LOW_LONG: f32 = 3.0;
const HOLE_HIGH_LONG: f32 = 2.0;
const HOLE_HIGH_LONG_LOW_RATE: f32 = 1.5;
/// Same, short windows.
const HOLE_LOW_SHORT: f32 = 2.0;
const HOLE_HIGH_SHORT: f32 = 1.5;
/// Bitrate per channel below which the upward hole slope is relaxed.
const LOW_RATE_THRESHOLD: u32 = 22_000;
/// Signal-to-mask ratio a band's own energy buys: -29 dB.
const SELF_MASKING_RATIO: f32 = 0.001_258_925;
/// Ceiling on a band's threshold, long windows.
const CLIP_ENERGY_LONG: f32 = 1.0e9;
/// Ceiling on a band's threshold, short windows.
const CLIP_ENERGY_SHORT: f32 = 15_625_000.0;
/// How much of a band's threshold survives pre-echo control.
const MIN_REMAINING_THRESHOLD: f32 = 0.01;
/// How far a threshold may rise above the previous frame's before pre-echo control
/// pulls it back.
const PRE_ECHO_RISE: f32 = 2.0;
/// A threshold high enough to disable pre-echo control for one frame.
const NO_PRE_ECHO: f32 = 1.0e20;

/// Absolute threshold of hearing, in dB, at each whole Bark.
///
/// Flat across the mid range where the ear is most sensitive, rising at both ends.
const BARK_QUIET_THRESHOLD: [f32; 25] = [
    15.0, 10.0, 7.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 3.0, 5.0, 10.0, 20.0, 30.0,
];

/// Level, in dB below a full-scale sine, at which the quiet threshold's floor sits.
///
/// The tables above are relative; something has to say where 0 dB of hearing
/// threshold falls against a digital full scale. The usual convention for a 16-bit
/// system puts full scale at about 96 dB SPL, which is what this is.
const FULL_SCALE_SPL_DB: f32 = 96.0;

/// What the model found for one frame.
#[derive(Debug, Clone)]
pub struct PsychoResult {
    /// Energy of each band.
    pub energy: [f32; MAX_BANDS],
    /// Noise energy each band can hide.
    pub threshold: [f32; MAX_BANDS],
    /// Energy spread across bands, for spotting bands that must not be zeroed.
    pub spread_energy: [f32; MAX_BANDS],
    /// Bits the frame needs to stay transparent, by the usual estimate.
    pub perceptual_entropy: f32,
    /// Bands the result covers.
    pub bands: usize,
}

impl Default for PsychoResult {
    fn default() -> Self {
        Self {
            energy: [0.0; MAX_BANDS],
            threshold: [0.0; MAX_BANDS],
            spread_energy: [0.0; MAX_BANDS],
            perceptual_entropy: 0.0,
            bands: 0,
        }
    }
}

/// The band layout and the constants derived from it.
///
/// All of this depends only on the sampling rate, the band table and the bitrate,
/// so it is computed once and reused for every frame.
#[derive(Debug, Clone)]
struct Layout {
    /// Bands the model covers.
    bands: usize,
    /// Lines in each band.
    width: Vec<usize>,
    /// Absolute threshold of hearing per band, as an energy.
    quiet: Vec<f32>,
    /// Masking of band `b` by `b - 1`, indexed by `b`.
    spread_up: Vec<f32>,
    /// Masking of band `b` by `b + 1`, indexed by `b`.
    spread_down: Vec<f32>,
    /// The same pair for the hole-detection energy.
    hole_up: Vec<f32>,
    hole_down: Vec<f32>,
    /// Floor on each band's signal-to-mask ratio.
    min_snr: Vec<f32>,
    clip_energy: f32,
}

/// The masking model for one channel.
#[derive(Debug, Clone)]
pub struct PsychoacousticModel {
    layout: Layout,
    /// Previous frame's thresholds, for pre-echo control.
    previous: Vec<f32>,
}

impl PsychoacousticModel {
    /// Build a model for a band table.
    ///
    /// `offsets` holds `bands + 1` line offsets, `bitrate_per_channel_bps` is what
    /// the rate loop will have to live within, and `short` selects the constants
    /// tuned for eight-short frames.
    pub fn new(
        sample_rate_hz: u32,
        bitrate_per_channel_bps: u32,
        offsets: &[usize],
        short: bool,
    ) -> Self {
        let bands = offsets.len().saturating_sub(1).min(MAX_BANDS);
        let lines = offsets[bands];
        let width: Vec<usize> = (0..bands).map(|b| offsets[b + 1] - offsets[b]).collect();

        // Bark value at the centre of each band, from the edges either side of it.
        let mut bark = Vec::with_capacity(bands);
        let mut previous_edge = 0.0f32;
        for b in 0..bands {
            let edge = bark_of_line(offsets[b + 1], lines, sample_rate_hz);
            bark.push(0.5 * (previous_edge + edge));
            previous_edge = edge;
        }

        let quiet = quiet_thresholds(&bark, &width, lines);
        let (spread_up, spread_down) = spreading(&bark, MASK_HIGH_FACTOR, MASK_LOW_FACTOR);
        let (hole_low, hole_high) = if short {
            (HOLE_LOW_SHORT, HOLE_HIGH_SHORT)
        } else if bitrate_per_channel_bps > LOW_RATE_THRESHOLD {
            (HOLE_LOW_LONG, HOLE_HIGH_LONG)
        } else {
            (HOLE_LOW_LONG, HOLE_HIGH_LONG_LOW_RATE)
        };
        let (hole_up, hole_down) = spreading(&bark, hole_high, hole_low);
        let min_snr =
            min_signal_to_mask(bitrate_per_channel_bps, sample_rate_hz, lines, offsets, &bark);

        Self {
            layout: Layout {
                bands,
                width,
                quiet,
                spread_up,
                spread_down,
                hole_up,
                hole_down,
                min_snr,
                clip_energy: if short { CLIP_ENERGY_SHORT } else { CLIP_ENERGY_LONG },
            },
            previous: vec![NO_PRE_ECHO; bands],
        }
    }

    /// Bands the model covers.
    #[inline]
    pub fn bands(&self) -> usize {
        self.layout.bands
    }

    /// Floor on band `band`'s signal-to-mask ratio.
    #[inline]
    pub fn min_snr(&self, band: usize) -> f32 {
        self.layout.min_snr.get(band).copied().unwrap_or(0.8)
    }

    /// Forget the previous frame, as at a discontinuity.
    pub fn reset(&mut self) {
        self.previous.fill(NO_PRE_ECHO);
    }

    /// Run the model over one transformed frame.
    ///
    /// `spectrum` is one window's MDCT coefficients and `offsets` the same band
    /// table the model was built for. `sequence` only matters at the two window
    /// transitions, where a long threshold must not be compared against a short one.
    pub fn analyse(
        &mut self,
        spectrum: &[f32],
        offsets: &[usize],
        sequence: WindowSequence,
        out: &mut PsychoResult,
    ) {
        let bands = self.layout.bands.min(offsets.len().saturating_sub(1));
        let mut energy = [0.0f32; MAX_BANDS];
        for b in 0..bands {
            let lo = offsets[b];
            let hi = offsets[b + 1].min(spectrum.len());
            let mut sum = 0.0f32;
            for &c in &spectrum[lo..hi] {
                sum += c * c;
            }
            energy[b] = sum;
        }
        self.analyse_energies(&energy[..bands], sequence, out);
    }

    /// Run the model over band energies that have already been measured.
    ///
    /// An eight-short frame codes each group of windows as one set of bands, so its
    /// energies come from several windows at once and cannot be read off a single
    /// spectrum; this is the entry point for that.
    pub fn analyse_energies(
        &mut self,
        energies: &[f32],
        sequence: WindowSequence,
        out: &mut PsychoResult,
    ) {
        let bands = self.layout.bands.min(energies.len());
        out.bands = bands;
        out.energy[..bands].copy_from_slice(&energies[..bands]);

        // A band masks itself down to -29 dB, no further than the clip.
        for b in 0..bands {
            out.threshold[b] = (out.energy[b] * SELF_MASKING_RATIO).min(self.layout.clip_energy);
        }

        spread_max(
            &mut out.threshold[..bands],
            &self.layout.spread_up,
            &self.layout.spread_down,
        );

        for b in 0..bands {
            out.threshold[b] = out.threshold[b].max(self.layout.quiet[b]);
        }

        // A long threshold measured either side of a window transition describes a
        // different span of time from the short ones next to it, so comparing them
        // would either over- or under-protect. Skip a frame instead.
        if sequence == WindowSequence::LongStopSequence {
            self.previous.fill(NO_PRE_ECHO);
        }
        for b in 0..bands {
            let ceiling = self.previous[b] * PRE_ECHO_RISE;
            let floor = MIN_REMAINING_THRESHOLD * out.threshold[b];
            self.previous[b] = out.threshold[b];
            out.threshold[b] = out.threshold[b].min(ceiling).max(floor);
        }
        if sequence == WindowSequence::LongStartSequence {
            self.previous.fill(NO_PRE_ECHO);
        }

        out.spread_energy[..bands].copy_from_slice(&out.energy[..bands]);
        spread_max(
            &mut out.spread_energy[..bands],
            &self.layout.hole_up,
            &self.layout.hole_down,
        );

        out.perceptual_entropy = perceptual_entropy(&out.energy, &out.threshold, &self.layout.width);
    }
}

/// Bark value of a spectral line.
///
/// The usual Zwicker fit, which turns a linear frequency axis into one where equal
/// distances are equally audible.
fn bark_of_line(line: usize, lines: usize, sample_rate_hz: u32) -> f32 {
    let centre = line as f32 * (sample_rate_hz as f32 * 0.5) / lines as f32;
    let temp = (1.333_333_3e-4 * centre).atan();
    13.3 * (0.000_76 * centre).atan() + 3.5 * temp * temp
}

/// Absolute threshold of hearing per band, as an energy in the encoder's units.
fn quiet_thresholds(bark: &[f32], width: &[usize], lines: usize) -> Vec<f32> {
    // Energy one line of a full-scale sine carries, which anchors the dB scale.
    // A sine of amplitude A windowed over 2N samples puts A * N into one MDCT bin.
    let full_scale = (32768.0f32 * lines as f32).powi(2);
    let floor = full_scale * 10f32.powf(-FULL_SCALE_SPL_DB / 10.0);

    (0..bark.len())
        .map(|b| {
            // The threshold applies across the band, so take the quieter of the two
            // Bark values bounding it rather than the centre.
            let lower = if b > 0 { 0.5 * (bark[b] + bark[b - 1]) } else { 0.5 * bark[b] };
            let upper = if b + 1 < bark.len() { 0.5 * (bark[b] + bark[b + 1]) } else { bark[b] };
            let db = quiet_db(lower).min(quiet_db(upper));
            floor * 10f32.powf(db / 10.0) * width[b] as f32
        })
        .collect()
}

/// Absolute threshold in dB at a Bark value, from the table.
fn quiet_db(bark: f32) -> f32 {
    let index = (bark.clamp(0.0, MAX_BARK)) as usize;
    BARK_QUIET_THRESHOLD[index.min(BARK_QUIET_THRESHOLD.len() - 1)]
}

/// Per-band spreading gains for a pair of slopes, in dB per Bark.
///
/// Returns `(up, down)`: `up[b]` is how much of band `b - 1` reaches `b`, and
/// `down[b]` how much of band `b + 1` reaches `b`.
fn spreading(bark: &[f32], high_slope: f32, low_slope: f32) -> (Vec<f32>, Vec<f32>) {
    let bands = bark.len();
    let mut up = vec![0.0f32; bands];
    let mut down = vec![0.0f32; bands];
    for b in 1..bands {
        let distance = bark[b] - bark[b - 1];
        up[b] = 10f32.powf(-high_slope * distance);
        down[b - 1] = 10f32.powf(-low_slope * distance);
    }
    (up, down)
}

/// Let each band raise its neighbours, in both directions.
///
/// A running maximum rather than a sum: the reference model takes the loudest
/// masker rather than adding them, which is both cheaper and closer to what the ear
/// does with a dominant tone.
fn spread_max(values: &mut [f32], up: &[f32], down: &[f32]) {
    for b in 1..values.len() {
        let spread = values[b - 1] * up[b];
        if values[b] < spread {
            values[b] = spread;
        }
    }
    for b in (0..values.len().saturating_sub(1)).rev() {
        let spread = values[b + 1] * down[b];
        if values[b] < spread {
            values[b] = spread;
        }
    }
}

/// Floor on each band's signal-to-mask ratio.
///
/// Left alone, the model would let a band with little energy of its own be masked
/// away entirely by its neighbours, which sounds worse than the bits it saves. The
/// floor is derived from the bits per Bark the target bitrate can afford, so a
/// generous bitrate protects more bands than a tight one.
fn min_signal_to_mask(
    bitrate_bps: u32,
    sample_rate_hz: u32,
    lines: usize,
    offsets: &[usize],
    bark: &[f32],
) -> Vec<f32> {
    let bands = bark.len();
    // Perceptual entropy the bitrate buys for one window, at 1.18 bits per unit.
    let pe_per_window = 1.18 * bitrate_bps as f32 / sample_rate_hz as f32 * lines as f32;
    let bark_scale = 1.0 / (bark[bands - 1] / MAX_BARK).min(1.0);

    let mut out = Vec::with_capacity(bands);
    let mut previous = 0.0f32;
    for b in 0..bands {
        let edge = 2.0 * bark[b] - previous;
        let bark_width = edge - previous;
        previous = edge;

        let width = (offsets[b + 1] - offsets[b]).max(1) as f32;
        let share = pe_per_window * 0.024 * bark_scale * bark_width / width;
        let snr = 2f32.powf(share) - 1.5;
        out.push((1.0 / snr.max(1.0)).clamp(0.003, 0.8));
    }
    out
}

/// Bits the frame needs to stay transparent.
///
/// Each band contributes the bits it takes to code a signal of its energy to a noise
/// floor at its threshold, which is what the rate loop compares against its budget.
fn perceptual_entropy(energy: &[f32], threshold: &[f32], width: &[usize]) -> f32 {
    let mut pe = 0.0f32;
    for b in 0..width.len() {
        let ratio = (energy[b] / threshold[b].max(f32::MIN_POSITIVE)).max(1.0);
        pe += width[b] as f32 * 0.5 * ratio.log2();
    }
    pe
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offsets(bands: usize, width: usize) -> Vec<usize> {
        (0..=bands).map(|b| b * width).collect()
    }

    /// Bark values must rise monotonically and reach the top of the scale.
    #[test]
    fn the_bark_scale_covers_the_audio_band() {
        let mut previous = -1.0;
        for line in (0..=1024).step_by(32) {
            let bark = bark_of_line(line, 1024, 44100);
            assert!(bark > previous, "bark went backwards at line {line}");
            previous = bark;
        }
        assert!(bark_of_line(1024, 1024, 44100) > 20.0);
    }

    /// A lone loud band must raise its neighbours, and more below it than above.
    #[test]
    fn spreading_is_asymmetric() {
        let bark: Vec<f32> = (0..8).map(|b| b as f32).collect();
        let (up, down) = spreading(&bark, MASK_HIGH_FACTOR, MASK_LOW_FACTOR);
        let mut values = vec![0.0f32; 8];
        values[4] = 1.0;
        spread_max(&mut values, &up, &down);

        assert!(values[5] > values[3], "spreading should favour higher bands");
        assert!(values[3] > 0.0 && values[5] > 0.0);
        assert!(values[5] < values[4] && values[3] < values[4]);
    }

    /// Silence must leave every threshold at the absolute threshold of hearing, and
    /// cost no perceptual entropy at all.
    #[test]
    fn silence_needs_no_bits() {
        let table = offsets(20, 16);
        let mut model = PsychoacousticModel::new(44100, 64000, &table, false);
        let mut out = PsychoResult::default();
        model.analyse(&vec![0.0; 320], &table, WindowSequence::OnlyLongSequence, &mut out);

        assert_eq!(out.perceptual_entropy, 0.0);
        for b in 0..out.bands {
            assert!(out.threshold[b] > 0.0, "band {b} has no threshold at all");
        }
    }

    /// A loud tone must cost bits, and more of them than a quiet one.
    #[test]
    fn perceptual_entropy_tracks_level() {
        let table = offsets(20, 16);
        let mut model = PsychoacousticModel::new(44100, 64000, &table, false);
        let mut out = PsychoResult::default();

        let mut quiet = vec![0.0f32; 320];
        quiet[40] = 1.0e4;
        model.analyse(&quiet, &table, WindowSequence::OnlyLongSequence, &mut out);
        let quiet_pe = out.perceptual_entropy;

        model.reset();
        let mut loud = vec![0.0f32; 320];
        loud[40] = 1.0e7;
        model.analyse(&loud, &table, WindowSequence::OnlyLongSequence, &mut out);

        assert!(out.perceptual_entropy > quiet_pe, "a louder tone must cost more bits");
        assert!(quiet_pe > 0.0);
    }

    /// An attack after a quiet passage must not be allowed to raise the threshold
    /// freely, or the noise the quantizer spreads over the whole window becomes
    /// audible in the quiet that preceded it.
    #[test]
    fn pre_echo_control_holds_a_threshold_down_after_quiet() {
        let table = offsets(20, 16);
        let loud = vec![1.0e6f32; 320];
        let quiet = vec![1.0f32; 320];

        // With no history the model has nothing to hold the threshold down.
        let mut unconstrained = PsychoacousticModel::new(44100, 64000, &table, false);
        let mut free = PsychoResult::default();
        unconstrained.analyse(&loud, &table, WindowSequence::OnlyLongSequence, &mut free);

        // Primed with a quiet frame, the same attack must be treated as riskier.
        let mut primed = PsychoacousticModel::new(44100, 64000, &table, false);
        let mut held = PsychoResult::default();
        primed.analyse(&quiet, &table, WindowSequence::OnlyLongSequence, &mut held);
        let settled: Vec<f32> = held.threshold[..held.bands].to_vec();
        primed.analyse(&loud, &table, WindowSequence::OnlyLongSequence, &mut held);

        for b in 0..held.bands {
            assert!(
                held.threshold[b] < free.threshold[b],
                "band {b} was not held down at all: {} vs {}",
                held.threshold[b],
                free.threshold[b]
            );
            // The floor keeps the threshold from collapsing to nothing.
            assert!(held.threshold[b] >= MIN_REMAINING_THRESHOLD * free.threshold[b] * 0.999);
            assert!(settled[b] > 0.0);
        }
    }
}
