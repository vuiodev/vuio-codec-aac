//! Dynamic range control.
//!
//! A broadcast or streaming encoder that expects its output to be heard in a car,
//! on a phone, or late at night can send, per frame, how much each part of the
//! spectrum should be turned up or down. The signal itself stays untouched, so a
//! listener with a quiet room gets the full dynamic range and everyone else gets a
//! flattened version of the same stream.
//!
//! What travels is `dynamic_range_info()` (ISO/IEC 14496-3 clause 4.5.2.7), inside a
//! fill element: up to sixteen frequency bands, each with a gain in quarter-decibel
//! steps, plus the *program reference level* the material was mastered against.
//!
//! # What the decoder decides
//!
//! Three things are the player's, not the stream's:
//!
//! * how much of a *cut* to honour, as a fraction — a listener who wants the full
//!   range sets it to zero,
//! * how much of a *boost* to honour, likewise,
//! * what output level to normalise to, against the transmitted reference.
//!
//! [`DrcSettings`] carries all three, and by default honours neither cut nor boost,
//! which leaves a stream carrying DRC decoding exactly as it would without it.

use crate::bitstream::BitReader;
use crate::error::Result;

pub mod channel_layout;
pub mod downmix_instructions;
pub mod gain_modifiers;
pub mod gain_set_params;
pub mod loudness_info;

/// Frequency bands one payload may divide the spectrum into.
pub const MAX_DRC_BANDS: usize = 16;
/// Spectral lines each unit of `drc_band_top` stands for.
const LINES_PER_BAND_UNIT: usize = 4;
/// Channels the exclusion mask can name.
const MAX_CHANNELS: usize = 8;
/// Reference level the standard assumes when a stream transmits none, in quarter
/// decibels below full scale.
const DEFAULT_REFERENCE_LEVEL: u8 = 108;
/// Decibels one step of a transmitted gain or reference level stands for.
const STEP_DB: f32 = 0.25;

/// What a listener wants done with the transmitted metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrcSettings {
    /// Fraction of a transmitted attenuation to apply, in `0.0..=1.0`.
    pub cut: f32,
    /// Fraction of a transmitted boost to apply, in `0.0..=1.0`.
    pub boost: f32,
    /// Level to normalise to, in quarter decibels below full scale, or `None` to
    /// leave the programme at the level it was mastered at.
    pub target_level: Option<u8>,
}

impl Default for DrcSettings {
    /// Honour nothing, which decodes the stream as though it carried no metadata.
    fn default() -> Self {
        Self { cut: 0.0, boost: 0.0, target_level: None }
    }
}

impl DrcSettings {
    /// Apply the transmitted range compression in full, without changing the level.
    pub fn full_compression() -> Self {
        Self { cut: 1.0, boost: 1.0, target_level: None }
    }

    /// Apply full compression and normalise to `target` quarter-decibels below full
    /// scale; -23 LUFS, the usual broadcast target, is about 92.
    pub fn normalised_to(target: u8) -> Self {
        Self { cut: 1.0, boost: 1.0, target_level: Some(target) }
    }
}

/// One frame's transmitted metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct DrcInfo {
    /// Bands the payload divides the spectrum into.
    pub bands: usize,
    /// Last spectral line of each band, exclusive.
    pub band_top: [usize; MAX_DRC_BANDS],
    /// Gain of each band, in quarter decibels; negative attenuates.
    pub gain: [i8; MAX_DRC_BANDS],
    /// Level the programme was mastered at, in quarter decibels below full scale.
    pub reference_level: Option<u8>,
    /// Whether each channel is subject to the metadata.
    pub channel_included: [bool; MAX_CHANNELS],
    /// How the decoder should move between this frame's gains and the last's.
    pub interpolation_scheme: u8,
}

impl Default for DrcInfo {
    fn default() -> Self {
        Self {
            bands: 1,
            band_top: [usize::MAX; MAX_DRC_BANDS],
            gain: [0; MAX_DRC_BANDS],
            reference_level: None,
            channel_included: [true; MAX_CHANNELS],
            interpolation_scheme: 0,
        }
    }
}

impl DrcInfo {
    /// Parse one `dynamic_range_info()` element.
    pub fn parse(reader: &mut BitReader) -> Result<Self> {
        let mut info = Self::default();

        if reader.read_bit()? {
            let _pce_instance_tag = reader.read_u8(4)?;
            let _reserved = reader.read_u8(4)?;
        }

        if reader.read_bit()? {
            read_excluded_channels(reader, &mut info.channel_included)?;
        }

        if reader.read_bit()? {
            info.bands = reader.read_u8(4)? as usize + 1;
            info.interpolation_scheme = reader.read_u8(4)?;
            for b in 0..info.bands {
                let top = reader.read_u8(8)? as usize;
                info.band_top[b] = (top + 1) * LINES_PER_BAND_UNIT;
            }
        } else {
            // One band covering everything.
            info.band_top[0] = usize::MAX;
        }

        if reader.read_bit()? {
            info.reference_level = Some(reader.read_u8(7)?);
            let _reserved = reader.read_bit()?;
        }

        for b in 0..info.bands {
            let negative = reader.read_bit()?;
            let magnitude = reader.read_u8(7)? as i32;
            info.gain[b] = if negative { -magnitude } else { magnitude } as i8;
        }

        Ok(info)
    }

    /// Whether any band asks for a level change.
    #[inline]
    pub fn is_neutral(&self) -> bool {
        self.gain[..self.bands].iter().all(|&g| g == 0)
    }
}

/// Read the `excluded_channels()` mask, which runs seven channels to a byte.
fn read_excluded_channels(reader: &mut BitReader, included: &mut [bool; MAX_CHANNELS]) -> Result<()> {
    let mut channel = 0usize;
    loop {
        for _ in 0..7 {
            let excluded = reader.read_bit()?;
            if channel < MAX_CHANNELS {
                included[channel] = !excluded;
            }
            channel += 1;
        }
        if !reader.read_bit()? {
            return Ok(());
        }
        // A stream may name more channels than any configuration has; the extra
        // flags still have to be read to stay aligned.
        if channel > 64 {
            return Ok(());
        }
    }
}

/// Applies transmitted dynamic range control to a decoded frame.
#[derive(Debug, Clone)]
pub struct DrcDecoder {
    settings: DrcSettings,
    /// The metadata in force, which persists until a frame carries new metadata.
    info: DrcInfo,
    /// Reference level last transmitted, which outlives the frame that carried it.
    reference_level: u8,
    /// Whether any payload has been seen.
    present: bool,
}

impl Default for DrcDecoder {
    fn default() -> Self {
        Self::new(DrcSettings::default())
    }
}

impl DrcDecoder {
    /// Build a decoder that will honour `settings`.
    pub fn new(settings: DrcSettings) -> Self {
        Self {
            settings,
            info: DrcInfo::default(),
            reference_level: DEFAULT_REFERENCE_LEVEL,
            present: false,
        }
    }

    /// Change what the listener asked for.
    pub fn set_settings(&mut self, settings: DrcSettings) {
        self.settings = settings;
    }

    /// What the listener asked for.
    #[inline]
    pub fn settings(&self) -> DrcSettings {
        self.settings
    }

    /// Whether any metadata has been seen in this stream.
    #[inline]
    pub fn is_present(&self) -> bool {
        self.present
    }

    /// The metadata in force.
    #[inline]
    pub fn info(&self) -> &DrcInfo {
        &self.info
    }

    /// Forget everything, as after a seek.
    pub fn reset(&mut self) {
        self.info = DrcInfo::default();
        self.reference_level = DEFAULT_REFERENCE_LEVEL;
        self.present = false;
    }

    /// Adopt one frame's metadata.
    pub fn accept(&mut self, info: DrcInfo) {
        if let Some(level) = info.reference_level {
            self.reference_level = level;
        }
        self.info = info;
        self.present = true;
    }

    /// Whether applying the current metadata would change anything.
    ///
    /// A stream carrying DRC but a listener honouring none of it should cost
    /// nothing, so the decode chain checks this before touching a frame.
    pub fn is_active(&self) -> bool {
        if !self.present {
            return false;
        }
        if self.normalisation_gain() != 1.0 {
            return true;
        }
        (0..self.info.bands).any(|b| self.band_gain(b) != 1.0)
    }

    /// Gain the level normalisation asks for, as a linear factor.
    fn normalisation_gain(&self) -> f32 {
        match self.settings.target_level {
            Some(target) => {
                let difference = target as f32 - self.reference_level as f32;
                10f32.powf(difference * STEP_DB / 20.0)
            }
            None => 1.0,
        }
    }

    /// Gain of one band, as a linear factor, after the listener's preferences.
    fn band_gain(&self, band: usize) -> f32 {
        let transmitted = self.info.gain[band] as f32;
        let fraction = if transmitted < 0.0 { self.settings.cut } else { self.settings.boost };
        10f32.powf(transmitted * STEP_DB * fraction.clamp(0.0, 1.0) / 20.0)
    }

    /// Apply the metadata to one channel's spectral coefficients.
    ///
    /// This is where the standard puts it: the gains are per frequency band, so
    /// applying them before the transform costs one multiply per coefficient and
    /// needs no filtering to keep the bands apart.
    pub fn apply_to_spectrum(&self, channel: usize, spectrum: &mut [f32]) {
        if !self.is_active() || !self.channel_included(channel) {
            return;
        }
        let normalisation = self.normalisation_gain();

        let mut start = 0usize;
        for band in 0..self.info.bands {
            let end = self.info.band_top[band].min(spectrum.len());
            if end <= start {
                start = end.max(start);
                continue;
            }
            let gain = self.band_gain(band) * normalisation;
            if gain != 1.0 {
                for value in &mut spectrum[start..end] {
                    *value *= gain;
                }
            }
            start = end;
        }
    }

    /// Apply the metadata to one channel's decoded samples.
    ///
    /// Used where band replication has already run: the replicated range is not in
    /// the core spectrum, so a per-band gain there would leave the top of the
    /// spectrum at the wrong level. The band covering the top of the core is the one
    /// the whole signal takes, which is exact for the single-band metadata that
    /// almost every stream carries.
    pub fn apply_to_samples(&self, channel: usize, samples: &mut [f32]) {
        if !self.is_active() || !self.channel_included(channel) {
            return;
        }
        let gain = self.band_gain(self.info.bands - 1) * self.normalisation_gain();
        if gain == 1.0 {
            return;
        }
        for value in samples {
            *value *= gain;
        }
    }

    /// Whether the metadata applies to a channel.
    #[inline]
    fn channel_included(&self, channel: usize) -> bool {
        self.info.channel_included.get(channel).copied().unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;

    /// Build a `dynamic_range_info()` payload for a test.
    fn payload(bands: &[(usize, i8)], reference: Option<u8>) -> Vec<u8> {
        let mut w = BitWriter::with_capacity(32);
        w.write_bit(false); // pce_tag_present
        w.write_bit(false); // excluded_chns_present
        if bands.len() > 1 {
            w.write_bit(true);
            w.write_u8(bands.len() as u8 - 1, 4);
            w.write_u8(0, 4); // interpolation scheme
            for &(top, _) in bands {
                w.write_u8((top / 4 - 1) as u8, 8);
            }
        } else {
            w.write_bit(false);
        }
        match reference {
            Some(level) => {
                w.write_bit(true);
                w.write_u8(level, 7);
                w.write_bit(false);
            }
            None => w.write_bit(false),
        }
        for &(_, gain) in bands {
            w.write_bit(gain < 0);
            w.write_u8(gain.unsigned_abs(), 7);
        }
        w.byte_align_zero();
        w.into_bytes()
    }

    /// Every field must come back exactly as it went in.
    #[test]
    fn a_payload_round_trips() {
        let bytes = payload(&[(256, -12), (1024, 8)], Some(96));
        let mut reader = BitReader::new(&bytes);
        let info = DrcInfo::parse(&mut reader).unwrap();

        assert_eq!(info.bands, 2);
        assert_eq!(info.band_top[0], 256);
        assert_eq!(info.band_top[1], 1024);
        assert_eq!(info.gain[0], -12);
        assert_eq!(info.gain[1], 8);
        assert_eq!(info.reference_level, Some(96));
    }

    /// A payload with no band field covers the whole spectrum with one gain.
    #[test]
    fn a_single_band_payload_covers_everything() {
        let bytes = payload(&[(0, -20)], None);
        let mut reader = BitReader::new(&bytes);
        let info = DrcInfo::parse(&mut reader).unwrap();

        assert_eq!(info.bands, 1);
        assert_eq!(info.gain[0], -20);

        let mut drc = DrcDecoder::new(DrcSettings::full_compression());
        drc.accept(info);
        let mut spectrum = vec![1.0f32; 1024];
        drc.apply_to_spectrum(0, &mut spectrum);

        // -20 quarter-decibels is -5 dB.
        let want = 10f32.powf(-5.0 / 20.0);
        for &v in &spectrum {
            assert!((v - want).abs() < 1e-6, "got {v}, want {want}");
        }
    }

    /// Honouring nothing must leave the signal untouched, whatever was transmitted.
    #[test]
    fn the_default_settings_change_nothing() {
        let bytes = payload(&[(0, -40)], Some(120));
        let mut reader = BitReader::new(&bytes);
        let mut drc = DrcDecoder::default();
        drc.accept(DrcInfo::parse(&mut reader).unwrap());

        assert!(!drc.is_active());
        let mut spectrum = vec![0.5f32; 64];
        drc.apply_to_spectrum(0, &mut spectrum);
        assert!(spectrum.iter().all(|&v| v == 0.5));
    }

    /// A partial cut must land partway between no compression and all of it.
    #[test]
    fn a_partial_cut_is_honoured_proportionally() {
        let bytes = payload(&[(0, -24)], None);
        let mut reader = BitReader::new(&bytes);
        let info = DrcInfo::parse(&mut reader).unwrap();

        let mut half = DrcDecoder::new(DrcSettings { cut: 0.5, boost: 0.5, target_level: None });
        half.accept(info.clone());
        let mut full = DrcDecoder::new(DrcSettings::full_compression());
        full.accept(info);

        let mut a = vec![1.0f32; 8];
        let mut b = vec![1.0f32; 8];
        half.apply_to_spectrum(0, &mut a);
        full.apply_to_spectrum(0, &mut b);

        // -24 quarter-decibels is -6 dB; half of it is -3 dB.
        assert!((a[0] - 10f32.powf(-3.0 / 20.0)).abs() < 1e-5, "{}", a[0]);
        assert!((b[0] - 10f32.powf(-6.0 / 20.0)).abs() < 1e-5, "{}", b[0]);
    }

    /// Normalising to a level below the transmitted reference must attenuate.
    #[test]
    fn level_normalisation_follows_the_reference() {
        let bytes = payload(&[(0, 0)], Some(100));
        let mut reader = BitReader::new(&bytes);
        let mut drc = DrcDecoder::new(DrcSettings::normalised_to(92));
        drc.accept(DrcInfo::parse(&mut reader).unwrap());

        let mut samples = vec![1.0f32; 8];
        drc.apply_to_samples(0, &mut samples);
        // Eight quarter-decibels quieter is -2 dB.
        assert!((samples[0] - 10f32.powf(-2.0 / 20.0)).abs() < 1e-5, "{}", samples[0]);
    }

    /// An excluded channel must be left alone.
    #[test]
    fn excluded_channels_are_left_alone() {
        let mut w = BitWriter::with_capacity(16);
        w.write_bit(false); // pce_tag_present
        w.write_bit(true); // excluded_chns_present
        // Channels 0 and 2 excluded, no continuation.
        for ch in 0..7 {
            w.write_bit(ch == 0 || ch == 2);
        }
        w.write_bit(false);
        w.write_bit(false); // drc_bands_present
        w.write_bit(false); // prog_ref_level_present
        w.write_bit(true);
        w.write_u8(20, 7); // -20 quarter-decibels
        w.byte_align_zero();
        let bytes = w.into_bytes();

        let mut reader = BitReader::new(&bytes);
        let mut drc = DrcDecoder::new(DrcSettings::full_compression());
        drc.accept(DrcInfo::parse(&mut reader).unwrap());

        for channel in 0..4 {
            let mut spectrum = vec![1.0f32; 8];
            drc.apply_to_spectrum(channel, &mut spectrum);
            let touched = spectrum[0] != 1.0;
            assert_eq!(touched, channel != 0 && channel != 2, "channel {channel}");
        }
    }
}
