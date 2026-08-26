//! Per-channel SBR payload: inverse filtering modes, envelopes, noise floors and
//! added sinusoids.
//!
//! Envelope and noise scalefactors are delta coded, either against the previous
//! band of the same envelope or against the same band of the previous envelope —
//! and the previous envelope may be the last one of the *previous frame*, which is
//! why [`ChannelHistory`] exists and why a decoder cannot start mid-stream without
//! a frame that codes its first envelope in the frequency direction.

use crate::bitstream::BitReader;
use crate::decoder::sbr::grid::{FrameClass, FrameGrid, MAX_NOISE_ENVELOPES};
use crate::decoder::sbr::header::{AmplitudeResolution, BandLayout, MAX_NOISE_BANDS, MAX_SFB};
use crate::error::{DecodeError, Result};
use crate::tables::sbr::{
    ENV_BALANCE_1_5_FREQ, ENV_BALANCE_1_5_TIME, ENV_BALANCE_3_0_FREQ, ENV_BALANCE_3_0_TIME,
    ENV_LEVEL_1_5_FREQ, ENV_LEVEL_1_5_TIME, ENV_LEVEL_3_0_FREQ, ENV_LEVEL_3_0_TIME,
    NOISE_BALANCE_3_0_TIME, NOISE_LEVEL_3_0_TIME, SbrCodebook,
};

/// Exponent offset the envelope dequantizer adds, `E = 2^(q * scale + 6)`.
const ENVELOPE_EXPONENT_OFFSET: f32 = 6.0;
/// Exponent the noise floor dequantizer subtracts from, `Q = 2^(6 - q)`.
const NOISE_FLOOR_OFFSET: f32 = 6.0;
/// Offset the coupled noise balance channel is coded around.
const NOISE_PAN_OFFSET: f32 = 12.0;

/// How strongly the inverse filter whitens a copied band.
///
/// Copying a low band upwards carries its tonal structure with it, which sounds
/// wrong where the original high band was noise-like. The inverse filter flattens
/// the copy by chirping the prediction filter towards the unit circle's interior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InverseFilterMode {
    /// No whitening.
    #[default]
    Off,
    /// Gentle whitening.
    Low,
    /// Moderate whitening.
    Mid,
    /// Strong whitening.
    High,
}

impl InverseFilterMode {
    fn from_bits(bits: u8) -> Self {
        match bits {
            0 => Self::Off,
            1 => Self::Low,
            2 => Self::Mid,
            _ => Self::High,
        }
    }

    /// Target chirp factor for this mode.
    #[inline]
    pub const fn chirp_target(self) -> f32 {
        match self {
            Self::Off => 0.0,
            Self::Low => 0.6,
            Self::Mid => 0.75,
            Self::High => 0.9,
        }
    }
}

/// What the delta-coded scalefactors of the previous frame were, so this frame's
/// time-direction deltas resolve.
#[derive(Debug, Clone, Default)]
pub struct ChannelHistory {
    /// Last envelope of the previous frame, at its own resolution.
    pub envelope: Vec<i32>,
    /// Whether that envelope used the high resolution.
    pub envelope_high_res: bool,
    /// Last noise floor of the previous frame.
    pub noise: Vec<i32>,
    /// Whether anything has been decoded yet.
    pub primed: bool,
}

impl ChannelHistory {
    /// Forget everything, as after a seek or a header change.
    pub fn reset(&mut self) {
        self.envelope.clear();
        self.noise.clear();
        self.envelope_high_res = false;
        self.primed = false;
    }
}

/// One channel's decoded SBR payload for one frame.
#[derive(Debug, Clone, Default)]
pub struct SbrChannelData {
    /// Time grid this channel's envelopes follow.
    pub grid: FrameGrid,
    /// Resolution the envelope scalefactors were coded at.
    pub amp_res: AmplitudeResolution,
    /// Inverse filtering mode per noise band.
    pub invf: Vec<InverseFilterMode>,
    /// Dequantized envelope energies, `[envelope][band]`.
    pub envelope: Vec<Vec<f32>>,
    /// Dequantized noise floors, `[noise envelope][band]`.
    pub noise: Vec<Vec<f32>>,
    /// Which high-resolution bands carry an added sinusoid.
    pub add_harmonic: Vec<bool>,
    /// Raw envelope scalefactors, kept so a coupled pair can be combined and so the
    /// next frame's time deltas have a reference.
    pub envelope_q: Vec<Vec<i32>>,
    /// Raw noise scalefactors, kept for the same reasons.
    pub noise_q: Vec<Vec<i32>>,
}

/// Read `sbr_dtdf()`: whether each envelope and noise floor is delta coded in time
/// rather than in frequency.
pub fn read_direction_flags(
    reader: &mut BitReader,
    grid: &FrameGrid,
) -> Result<(Vec<bool>, Vec<bool>)> {
    let env: Vec<bool> =
        (0..grid.envelopes()).map(|_| reader.read_bit()).collect::<Result<_>>()?;
    let noise: Vec<bool> =
        (0..grid.noise_envelopes()).map(|_| reader.read_bit()).collect::<Result<_>>()?;
    Ok((env, noise))
}

/// Read `sbr_invf()`, one two-bit mode per noise band.
pub fn read_invf(reader: &mut BitReader, bands: usize) -> Result<Vec<InverseFilterMode>> {
    (0..bands)
        .map(|_| Ok(InverseFilterMode::from_bits(reader.read_u8(2)?)))
        .collect()
}

/// The amplitude resolution a frame actually uses.
///
/// A frame with a single evenly spaced envelope always uses the fine resolution,
/// whatever the header said, because there is only one envelope to spend bits on.
pub fn effective_resolution(grid: &FrameGrid, header: AmplitudeResolution) -> AmplitudeResolution {
    if grid.class == FrameClass::FixFix && grid.envelopes() == 1 {
        AmplitudeResolution::Fine
    } else {
        header
    }
}

/// The four codebooks a channel uses, chosen by resolution and by whether this
/// channel carries balances rather than levels.
struct Books {
    env_time: &'static SbrCodebook,
    env_freq: &'static SbrCodebook,
    noise_time: &'static SbrCodebook,
    noise_freq: &'static SbrCodebook,
    /// Bits in the absolute value that opens a frequency-coded envelope.
    env_start_bits: usize,
    /// Scale applied to every decoded value; balances are coded at half rate.
    step: i32,
}

impl Books {
    fn select(amp_res: AmplitudeResolution, balance: bool) -> Self {
        match (amp_res, balance) {
            (AmplitudeResolution::Coarse, false) => Books {
                env_time: &ENV_LEVEL_3_0_TIME,
                env_freq: &ENV_LEVEL_3_0_FREQ,
                noise_time: &NOISE_LEVEL_3_0_TIME,
                noise_freq: &ENV_LEVEL_3_0_FREQ,
                env_start_bits: 6,
                step: 1,
            },
            (AmplitudeResolution::Fine, false) => Books {
                env_time: &ENV_LEVEL_1_5_TIME,
                env_freq: &ENV_LEVEL_1_5_FREQ,
                noise_time: &NOISE_LEVEL_3_0_TIME,
                noise_freq: &ENV_LEVEL_3_0_FREQ,
                env_start_bits: 7,
                step: 1,
            },
            (AmplitudeResolution::Coarse, true) => Books {
                env_time: &ENV_BALANCE_3_0_TIME,
                env_freq: &ENV_BALANCE_3_0_FREQ,
                noise_time: &NOISE_BALANCE_3_0_TIME,
                noise_freq: &ENV_BALANCE_3_0_FREQ,
                env_start_bits: 5,
                step: 2,
            },
            (AmplitudeResolution::Fine, true) => Books {
                env_time: &ENV_BALANCE_1_5_TIME,
                env_freq: &ENV_BALANCE_1_5_FREQ,
                noise_time: &NOISE_BALANCE_3_0_TIME,
                noise_freq: &ENV_BALANCE_3_0_FREQ,
                env_start_bits: 6,
                step: 2,
            },
        }
    }
}

/// Read `sbr_envelope()`, resolving the delta coding as it goes.
///
/// `balance` says this channel carries balances rather than levels, which selects
/// the narrower codebooks and doubles every decoded step.
pub fn read_envelopes(
    reader: &mut BitReader,
    layout: &BandLayout,
    grid: &FrameGrid,
    amp_res: AmplitudeResolution,
    balance: bool,
    time_delta: &[bool],
    history: &mut ChannelHistory,
) -> Result<Vec<Vec<i32>>> {
    let low_bands = layout.sfb_count(false);
    let high_bands = layout.sfb_count(true);
    if high_bands > MAX_SFB {
        return Err(corrupt("SBR band layout is out of range for this frame"));
    }
    let books = Books::select(amp_res, balance);
    let odd = high_bands & 1;

    let mut out: Vec<Vec<i32>> = Vec::with_capacity(grid.envelopes());
    for (l, &in_time) in time_delta.iter().enumerate() {
        let high_res = grid.high_res[l];
        let bands = if high_res { high_bands } else { low_bands };
        let mut row = vec![0i32; bands];

        if in_time {
            // Reference the previous envelope, which for the first one is the last
            // envelope of the previous frame.
            let (reference, reference_high_res) = if l == 0 {
                (history.envelope.as_slice(), history.envelope_high_res)
            } else {
                (out[l - 1].as_slice(), grid.high_res[l - 1])
            };
            if reference.is_empty() {
                return Err(corrupt("SBR envelope refers back to a frame that was not decoded"));
            }
            for (j, slot) in row.iter_mut().enumerate() {
                let k = map_band(j, high_res, reference_high_res, odd).min(reference.len() - 1);
                *slot = reference[k] + books.step * books.env_time.decode(reader)?;
            }
        } else {
            row[0] = books.step * reader.read_bits(books.env_start_bits)? as i32;
            for j in 1..bands {
                row[j] = row[j - 1] + books.step * books.env_freq.decode(reader)?;
            }
        }
        out.push(row);
    }

    history.envelope = out.last().cloned().unwrap_or_default();
    history.envelope_high_res = *grid.high_res.last().unwrap();
    history.primed = true;
    Ok(out)
}

/// Read `sbr_noise()`.
pub fn read_noise_floors(
    reader: &mut BitReader,
    layout: &BandLayout,
    amp_res: AmplitudeResolution,
    balance: bool,
    time_delta: &[bool],
    history: &mut ChannelHistory,
) -> Result<Vec<Vec<i32>>> {
    let noise_bands = layout.noise_band_count();
    if noise_bands > MAX_NOISE_BANDS {
        return Err(corrupt("SBR band layout asks for too many noise bands"));
    }
    if time_delta.len() > MAX_NOISE_ENVELOPES {
        return Err(corrupt("SBR frame carries too many noise envelopes"));
    }
    let books = Books::select(amp_res, balance);

    let mut out: Vec<Vec<i32>> = Vec::with_capacity(time_delta.len());
    for (l, &in_time) in time_delta.iter().enumerate() {
        let mut row = vec![0i32; noise_bands];
        if in_time {
            let reference = if l == 0 { history.noise.as_slice() } else { out[l - 1].as_slice() };
            if reference.len() != noise_bands {
                return Err(corrupt("SBR noise floor refers back to a different band layout"));
            }
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = reference[j] + books.step * books.noise_time.decode(reader)?;
            }
        } else {
            row[0] = books.step * reader.read_bits(5)? as i32;
            for j in 1..noise_bands {
                row[j] = row[j - 1] + books.step * books.noise_freq.decode(reader)?;
            }
        }
        out.push(row);
    }

    history.noise = out.last().cloned().unwrap_or_default();
    Ok(out)
}

/// Read `sbr_sinusoidal_coding()`, one flag per high-resolution band.
pub fn read_added_sinusoids(reader: &mut BitReader, high_bands: usize) -> Result<Vec<bool>> {
    (0..high_bands).map(|_| reader.read_bit()).collect()
}

/// Map band `j` at one resolution onto the band covering it at another.
///
/// The two resolutions nest — every low-resolution border is also a
/// high-resolution border — so the mapping is a halving or a doubling, offset by
/// one when the high-resolution band count is odd because the odd band out sits at
/// the bottom of the range.
#[inline]
fn map_band(j: usize, to_high_res: bool, from_high_res: bool, odd: usize) -> usize {
    match (to_high_res, from_high_res) {
        (a, b) if a == b => j,
        (true, false) => (j + odd) >> 1,
        (false, true) => {
            if j == 0 {
                0
            } else {
                2 * j - odd
            }
        }
        _ => unreachable!(),
    }
}

/// Turn one channel's scalefactors into linear energies.
pub fn dequantize(data: &mut SbrChannelData) {
    let scale = data.amp_res.dequant_scale();
    data.envelope = data
        .envelope_q
        .iter()
        .map(|row| {
            row.iter().map(|&q| exp2(q as f32 * scale + ENVELOPE_EXPONENT_OFFSET)).collect()
        })
        .collect();
    data.noise = data
        .noise_q
        .iter()
        .map(|row| row.iter().map(|&q| exp2(NOISE_FLOOR_OFFSET - q as f32)).collect())
        .collect();
}

/// Turn a coupled pair's level and balance scalefactors into per-channel energies.
///
/// `level` carries the pair's total and `balance` how it divides between them, so
/// neither channel's energy can be recovered without the other's scalefactors.
pub fn dequantize_coupled(level: &mut SbrChannelData, balance: &mut SbrChannelData) {
    let scale = level.amp_res.dequant_scale();
    let pan = level.amp_res.pan_offset();

    level.envelope.clear();
    balance.envelope.clear();
    for (l_row, b_row) in level.envelope_q.iter().zip(balance.envelope_q.iter()) {
        let mut left = Vec::with_capacity(l_row.len());
        let mut right = Vec::with_capacity(l_row.len());
        for (&lq, &bq) in l_row.iter().zip(b_row.iter()) {
            let total = exp2(lq as f32 * scale + 1.0 + ENVELOPE_EXPONENT_OFFSET);
            let tilt = (bq as f32 - pan) * scale;
            left.push(total / (1.0 + exp2(-tilt)));
            right.push(total / (1.0 + exp2(tilt)));
        }
        level.envelope.push(left);
        balance.envelope.push(right);
    }

    level.noise.clear();
    balance.noise.clear();
    for (l_row, b_row) in level.noise_q.iter().zip(balance.noise_q.iter()) {
        let mut left = Vec::with_capacity(l_row.len());
        let mut right = Vec::with_capacity(l_row.len());
        for (&lq, &bq) in l_row.iter().zip(b_row.iter()) {
            let total = exp2(NOISE_FLOOR_OFFSET - lq as f32 + 1.0);
            let tilt = bq as f32 - NOISE_PAN_OFFSET;
            left.push(total / (1.0 + exp2(-tilt)));
            right.push(total / (1.0 + exp2(tilt)));
        }
        level.noise.push(left);
        balance.noise.push(right);
    }
}

/// `2^x`, clamped so a corrupt scalefactor cannot produce an infinity that then
/// poisons every gain computed from it.
#[inline]
fn exp2(x: f32) -> f32 {
    x.clamp(-120.0, 120.0).exp2()
}

fn corrupt(message: &str) -> crate::error::Error {
    DecodeError::CorruptedFrame(message.into()).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mapping between the two resolutions must be a left inverse in the direction
    /// that loses no information.
    #[test]
    fn band_mapping_nests() {
        for high_bands in 1..24usize {
            let odd = high_bands & 1;
            let low_bands = high_bands.div_ceil(2);
            for j in 0..low_bands {
                let k = map_band(j, false, true, odd);
                assert!(k < high_bands, "low band {j} maps outside the high table");
            }
            for j in 0..high_bands {
                let k = map_band(j, true, false, odd);
                assert!(k < low_bands, "high band {j} maps outside the low table");
            }
        }
    }

    /// A coupled pair must divide the transmitted total between its channels.
    #[test]
    fn coupled_energies_sum_to_the_transmitted_total() {
        let mut level = SbrChannelData {
            amp_res: AmplitudeResolution::Coarse,
            envelope_q: vec![vec![20, 25]],
            noise_q: vec![vec![3, 4]],
            ..Default::default()
        };
        let mut balance = SbrChannelData {
            amp_res: AmplitudeResolution::Coarse,
            envelope_q: vec![vec![12, 16]],
            noise_q: vec![vec![12, 14]],
            ..Default::default()
        };
        dequantize_coupled(&mut level, &mut balance);

        for band in 0..2 {
            let total = level.envelope[0][band] + balance.envelope[0][band];
            let expected = (level.envelope_q[0][band] as f32 + 1.0 + 6.0).exp2();
            assert!(
                (total - expected).abs() < expected * 1e-5,
                "band {band}: {total} != {expected}"
            );
        }
        // A balance of exactly the pan offset splits the pair evenly.
        let mut even_level = SbrChannelData {
            amp_res: AmplitudeResolution::Coarse,
            envelope_q: vec![vec![10]],
            noise_q: vec![vec![2]],
            ..Default::default()
        };
        let mut even_balance = SbrChannelData {
            amp_res: AmplitudeResolution::Coarse,
            envelope_q: vec![vec![12]],
            noise_q: vec![vec![12]],
            ..Default::default()
        };
        dequantize_coupled(&mut even_level, &mut even_balance);
        assert!(
            (even_level.envelope[0][0] - even_balance.envelope[0][0]).abs() < 1e-3,
            "a centred balance must split evenly"
        );
    }

    /// Independent dequantization must follow the standard's exponent rule.
    #[test]
    fn independent_dequantization_matches_the_definition() {
        let mut data = SbrChannelData {
            amp_res: AmplitudeResolution::Fine,
            envelope_q: vec![vec![0, 2, -4]],
            noise_q: vec![vec![0, 6, -2]],
            ..Default::default()
        };
        dequantize(&mut data);
        assert_eq!(data.envelope[0], vec![64.0, 128.0, 16.0]);
        assert_eq!(data.noise[0], vec![64.0, 1.0, 256.0]);
    }

    /// Chirp targets must rise with the mode.
    #[test]
    fn chirp_targets_are_ordered() {
        let modes = [
            InverseFilterMode::Off,
            InverseFilterMode::Low,
            InverseFilterMode::Mid,
            InverseFilterMode::High,
        ];
        assert!(modes.windows(2).all(|w| w[0].chirp_target() < w[1].chirp_target()));
        assert!(modes.iter().all(|m| m.chirp_target() < 1.0));
    }
}
