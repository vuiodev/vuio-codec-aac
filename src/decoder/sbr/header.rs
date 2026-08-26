//! The SBR header and the band layout it determines.
//!
//! A header is small — a dozen fields — but almost everything downstream is
//! derived from it: which QMF subband the replicated range starts at, how the
//! range is divided into scalefactor bands at two resolutions, where the noise
//! floors and the gain limiter place their boundaries, and which low bands are
//! copied to which high bands. All of that derivation lives here, in
//! [`BandLayout::derive`], and it runs once per header rather than once per frame.

use crate::bitstream::BitReader;
use crate::error::{DecodeError, Result};

/// Subbands the synthesis filterbank runs, and so the highest band SBR may fill.
pub const SBR_SYNTHESIS_BANDS: usize = 64;
/// Most scalefactor bands the replicated range may be split into.
pub const MAX_SFB: usize = 48;
/// Most noise floor bands a header may ask for.
pub const MAX_NOISE_BANDS: usize = 5;
/// Most copy-up patches the band layout may need.
pub const MAX_PATCHES: usize = 6;
/// Subbands left between the end of one patch and the start of the next.
///
/// Zero in the base specification; kept named because the patch arithmetic reads
/// as arbitrary without it.
const GUARD_BANDS: usize = 0;
/// Lowest source subband a patch may copy from.
///
/// Band 0 carries DC and is never a useful source.
const FIRST_SOURCE_BAND: usize = 1;

/// Amplitude resolution of the transmitted envelope scalefactors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AmplitudeResolution {
    /// 1.5 dB steps, the finer grid, used for stationary material.
    Fine,
    /// 3 dB steps, which halves the envelope bitrate.
    #[default]
    Coarse,
}

impl AmplitudeResolution {
    /// Exponent scale applied when dequantizing: `E = 2^(q * scale + 6)`.
    #[inline]
    pub const fn dequant_scale(self) -> f32 {
        match self {
            Self::Fine => 0.5,
            Self::Coarse => 1.0,
        }
    }

    /// Offset the balance channel of a coupled pair is coded around.
    #[inline]
    pub const fn pan_offset(self) -> f32 {
        match self {
            Self::Fine => 24.0,
            Self::Coarse => 12.0,
        }
    }
}

/// The `sbr_header()` payload of ISO/IEC 14496-3, 4.5.2.8.1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SbrHeader {
    /// Resolution of the envelope scalefactors. A frame may override this.
    pub amp_res: AmplitudeResolution,
    /// Index into the per-rate table that sets the first replicated subband.
    pub start_freq: u8,
    /// Index that sets the last replicated subband.
    pub stop_freq: u8,
    /// How many master bands sit below the replicated range.
    pub xover_band: u8,
    /// Warping of the master band layout: 0 is linear, 1 to 3 are increasingly
    /// coarse logarithmic scales.
    pub freq_scale: u8,
    /// Stretches the upper region of a logarithmic layout.
    pub alter_scale: bool,
    /// Noise floor bands per octave, as an index.
    pub noise_bands: u8,
    /// Limiter bands per octave, as an index. Zero means one band overall.
    pub limiter_bands: u8,
    /// How far the limiter lets a band's gain exceed the band average.
    pub limiter_gains: u8,
    /// Whether the current energy is measured per subband or per scalefactor band.
    pub interpol_freq: bool,
    /// Whether gains are smoothed across envelope boundaries.
    pub smoothing_mode: bool,
}

impl Default for SbrHeader {
    fn default() -> Self {
        Self {
            amp_res: AmplitudeResolution::Coarse,
            start_freq: 5,
            stop_freq: 0,
            xover_band: 0,
            freq_scale: 2,
            alter_scale: true,
            noise_bands: 2,
            limiter_bands: 2,
            limiter_gains: 2,
            interpol_freq: true,
            smoothing_mode: true,
        }
    }
}

impl SbrHeader {
    /// Parse `sbr_header()`.
    ///
    /// The two extra-header flags gate fields that otherwise take the defaults the
    /// standard names, which is why the absent case is not simply "leave alone":
    /// a header that omits them resets those fields.
    pub fn parse(reader: &mut BitReader) -> Result<Self> {
        let amp_res = if reader.read_bit()? {
            AmplitudeResolution::Coarse
        } else {
            AmplitudeResolution::Fine
        };
        let start_freq = reader.read_u8(4)?;
        let stop_freq = reader.read_u8(4)?;
        let xover_band = reader.read_u8(3)?;
        let _reserved = reader.read_u8(2)?;
        let extra_1 = reader.read_bit()?;
        let extra_2 = reader.read_bit()?;

        let mut header = Self { amp_res, start_freq, stop_freq, xover_band, ..Self::default() };

        if extra_1 {
            header.freq_scale = reader.read_u8(2)?;
            header.alter_scale = reader.read_bit()?;
            header.noise_bands = reader.read_u8(2)?;
        }
        if extra_2 {
            header.limiter_bands = reader.read_u8(2)?;
            header.limiter_gains = reader.read_u8(2)?;
            header.interpol_freq = reader.read_bit()?;
            header.smoothing_mode = reader.read_bit()?;
        }
        Ok(header)
    }

    /// Whether a change from `other` leaves the band layout untouched.
    ///
    /// A header repeated unchanged must not reset the decoder's inter-frame state,
    /// so the comparison has to be exact rather than "close enough".
    pub fn same_layout_as(&self, other: &Self) -> bool {
        self.start_freq == other.start_freq
            && self.stop_freq == other.stop_freq
            && self.xover_band == other.xover_band
            && self.freq_scale == other.freq_scale
            && self.alter_scale == other.alter_scale
            && self.noise_bands == other.noise_bands
            && self.limiter_bands == other.limiter_bands
    }
}

/// One copy-up: a run of source subbands and where they land.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Patch {
    /// First subband the patch writes.
    pub dst_start: usize,
    /// First subband the patch reads.
    pub src_start: usize,
    /// How many subbands the patch covers.
    pub width: usize,
    /// Where the patch's guard region begins, which the limiter uses as a border.
    pub guard_start: usize,
}

impl Patch {
    /// Subband the patch reads for output subband `k`.
    #[inline]
    pub const fn source_of(&self, k: usize) -> usize {
        k - self.dst_start + self.src_start
    }
}

/// Every band boundary the rest of the SBR chain needs.
#[derive(Debug, Clone, Default)]
pub struct BandLayout {
    /// Master band boundaries, in subbands.
    pub master: Vec<u8>,
    /// Scalefactor band boundaries at low frequency resolution.
    pub low: Vec<u8>,
    /// Scalefactor band boundaries at high frequency resolution.
    pub high: Vec<u8>,
    /// Noise floor band boundaries.
    pub noise: Vec<u8>,
    /// Limiter band boundaries, relative to [`Self::k_x`].
    pub limiter: Vec<u8>,
    /// First replicated subband; below it the core signal passes through.
    pub k_x: usize,
    /// One past the last replicated subband.
    pub k_high: usize,
    /// Copy-ups feeding the replicated range.
    pub patches: Vec<Patch>,
}

impl BandLayout {
    /// Subbands SBR reconstructs.
    #[inline]
    pub fn replicated_bands(&self) -> usize {
        self.k_high - self.k_x
    }

    /// Scalefactor bands at the given resolution.
    #[inline]
    pub fn sfb_count(&self, high_res: bool) -> usize {
        if high_res { self.high.len() - 1 } else { self.low.len() - 1 }
    }

    /// Boundaries at the given resolution.
    #[inline]
    pub fn sfb_table(&self, high_res: bool) -> &[u8] {
        if high_res { &self.high } else { &self.low }
    }

    /// Noise floor bands.
    #[inline]
    pub fn noise_band_count(&self) -> usize {
        self.noise.len() - 1
    }

    /// Derive the whole layout from a header and the SBR output sampling rate.
    ///
    /// `output_rate_hz` is the rate *after* SBR, twice the core rate.
    pub fn derive(header: &SbrHeader, output_rate_hz: u32) -> Result<Self> {
        let (k0, k2) = start_and_stop_bands(header, output_rate_hz)?;
        let master = master_band_table(header, k0, k2)?;

        let xover = header.xover_band as usize;
        if xover >= master.len() {
            return Err(bad("SBR crossover band is past the end of the master table"));
        }

        let (low, high) = split_resolutions(&master, xover);
        let k_x = low[0] as usize;
        let k_high = *low.last().unwrap() as usize;
        if k_x >= k_high || k_x > 32 {
            return Err(bad("SBR replicated range is empty or starts above the core"));
        }

        let noise = noise_band_table(header, &low)?;
        let patches = build_patches(&master, k_x, k_high, output_rate_hz)?;
        let limiter = limiter_band_table(header, &low, &patches);

        Ok(Self { master, low, high, noise, limiter, k_x, k_high, patches })
    }
}

fn bad(message: &str) -> crate::error::Error {
    DecodeError::CorruptedFrame(message.into()).into()
}

/// Map a sampling rate to the one whose offset table the standard uses for it.
fn mapped_rate(rate_hz: u32) -> Result<u32> {
    Ok(match rate_hz {
        0..=18782 => 16000,
        18783..=23003 => 22050,
        23004..=27712 => 24000,
        27713..=35776 => 32000,
        35777..=41999 => 40000,
        42000..=46008 => 44100,
        46009..=55425 => 48000,
        55426..=75131 => 64000,
        75132..=92016 => 88200,
        _ => 96000,
    })
}

/// Offsets applied to the minimum start band, per mapped rate.
fn start_offsets(mapped: u32) -> &'static [i32; 16] {
    match mapped {
        16000 => &[-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
        22050 => &[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13],
        24000 => &[-5, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16],
        32000 => &[-6, -4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16],
        40000 => &[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19],
        44100 | 48000 | 64000 => &[-4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16, 20],
        88200 | 96000 => &[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16, 20, 24],
        _ => &[0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16, 20, 24, 28, 33],
    }
}

/// The first and last subband of the replicated range, `k0` and `k2`.
fn start_and_stop_bands(header: &SbrHeader, output_rate_hz: u32) -> Result<(usize, usize)> {
    let mapped = mapped_rate(output_rate_hz)?;

    // The lowest subband whose centre clears the anchor frequency for this rate.
    let anchor = if mapped < 32000 {
        3000.0
    } else if mapped < 64000 {
        4000.0
    } else {
        5000.0
    };
    let k0_min = (anchor * 2.0 * 64.0 / mapped as f64 + 0.5) as i32;
    let k0 = k0_min + start_offsets(mapped)[header.start_freq as usize];

    let stop_anchor = if output_rate_hz < 32000 {
        6000.0
    } else if output_rate_hz < 64000 {
        8000.0
    } else {
        10000.0
    };
    let k1_min = (stop_anchor * 2.0 * 64.0 / output_rate_hz as f64 + 0.5) as i32;

    let k2 = match header.stop_freq {
        // The table's own entry: a geometric run from `k1_min` up to 64, its
        // per-step widths sorted so the bands widen monotonically.
        s if s < 14 => {
            let mut steps = [0i32; 13];
            let mut previous = k1_min;
            for (i, step) in steps.iter_mut().enumerate() {
                let next =
                    (k1_min as f64 * (64.0 / k1_min as f64).powf((i + 1) as f64 / 13.0) + 0.5)
                        as i32;
                *step = next - previous;
                previous = next;
            }
            steps.sort_unstable();
            k1_min + steps[..s as usize].iter().sum::<i32>()
        }
        14 => 2 * k0,
        _ => 3 * k0,
    };

    let k0 = k0.clamp(0, SBR_SYNTHESIS_BANDS as i32) as usize;
    let k2 = k2.clamp(0, SBR_SYNTHESIS_BANDS as i32) as usize;

    if k2 <= k0 || k2 - k0 > MAX_SFB {
        return Err(bad("SBR replicated range is empty or wider than the standard allows"));
    }
    Ok((k0, k2))
}

/// Widths of `num_bands` bands spanning `start..stop` geometrically.
///
/// The widths come out in ascending order after sorting, which is what makes the
/// resulting boundaries a warped rather than merely geometric scale.
fn geometric_widths(start: usize, stop: usize, num_bands: usize) -> Vec<u8> {
    let ratio = (start as f64 / stop as f64).powf(1.0 / num_bands as f64);
    let mut widths = vec![0u8; num_bands];
    let mut previous = stop as i32;
    let mut exact = stop as f64;
    for slot in widths.iter_mut().rev() {
        exact *= ratio;
        let current = (exact + 0.5).floor() as i32;
        *slot = (previous - current).max(0) as u8;
        previous = current;
    }
    widths.sort_unstable();
    widths
}

/// Build the master band table.
fn master_band_table(header: &SbrHeader, k0: usize, k2: usize) -> Result<Vec<u8>> {
    if header.freq_scale == 0 {
        return linear_master_table(header, k0, k2);
    }

    let bands_per_octave = match header.freq_scale {
        1 => 12.0f64,
        2 => 10.0,
        _ => 8.0,
    };

    // Above roughly one and a bit octaves the standard splits the range in two so
    // the lower octave keeps full resolution and the upper one can be warped.
    let two_regions = 10000 * k2 > 22449 * k0;
    let k1 = if two_regions { 2 * k0 } else { k2 };

    let num_bands_0 = 2 * ((bands_per_octave * log2(k1, k0) / 2.0).round() as i32);
    if num_bands_0 < 1 {
        return Err(bad("SBR master table would have no bands in its lower region"));
    }
    let mut widths_0 = geometric_widths(k0, k1, num_bands_0 as usize);

    let mut table = Vec::with_capacity(MAX_SFB + 1);
    table.push(k0 as u8);
    let mut edge = k0;
    for w in &widths_0 {
        edge += *w as usize;
        table.push(edge as u8);
    }

    if !two_regions {
        if widths_0.first() == Some(&0) {
            return Err(bad("SBR master table has a zero-width band"));
        }
        return Ok(table);
    }

    let warp = if header.alter_scale { 1.3f64 } else { 1.0 };
    let num_bands_1 = 2 * ((bands_per_octave * log2(k2, k1) / (2.0 * warp)).round() as i32);
    if num_bands_1 < 1 {
        return Err(bad("SBR master table would have no bands in its upper region"));
    }
    let mut widths_1 = geometric_widths(k1, k2, num_bands_1 as usize);

    // Where the upper region would start narrower than the lower region ended, move
    // width from its widest band to its narrowest so the scale stays monotone.
    if widths_1[0] < widths_0[widths_0.len() - 1] {
        let change =
            (widths_0[widths_0.len() - 1] - widths_1[0]).min((widths_1[num_bands_1 as usize - 1] - widths_1[0]) / 2);
        widths_1[0] += change;
        let last = widths_1.len() - 1;
        widths_1[last] -= change;
        widths_1.sort_unstable();
    }

    let mut edge = k1;
    for w in &widths_1 {
        edge += *w as usize;
        table.push(edge as u8);
    }
    widths_0.clear();
    Ok(table)
}

/// Build the master band table for `bs_freq_scale == 0`, a linear scale.
fn linear_master_table(header: &SbrHeader, k0: usize, k2: usize) -> Result<Vec<u8>> {
    let (step, num_bands) = if header.alter_scale {
        (2usize, ((k2 - k0 + 2) >> 2) << 1)
    } else {
        (1usize, (k2 - k0) & !1)
    };
    if num_bands < 1 {
        return Err(bad("SBR linear master table would have no bands"));
    }

    let mut widths = vec![step as i32; num_bands];
    // Spread the rounding error of the uniform step over the outermost bands.
    let mut shortfall = k2 as i32 - (k0 + num_bands * step) as i32;
    let mut at = if shortfall < 0 { 0usize } else { num_bands - 1 };
    let direction: i32 = if shortfall < 0 { 1 } else { -1 };
    while shortfall != 0 {
        widths[at] -= direction;
        at = (at as i32 + direction) as usize;
        shortfall += direction;
    }

    let mut table = Vec::with_capacity(num_bands + 1);
    table.push(k0 as u8);
    let mut edge = k0 as i32;
    for w in widths {
        edge += w;
        table.push(edge as u8);
    }
    Ok(table)
}

/// `log2(a / b)`, the shape every band count in this module is built from.
#[inline]
fn log2(a: usize, b: usize) -> f64 {
    (a as f64 / b as f64).log2()
}

/// Split the master table above the crossover into the two scalefactor
/// resolutions: high takes every master band, low takes every second one.
fn split_resolutions(master: &[u8], xover: usize) -> (Vec<u8>, Vec<u8>) {
    let high: Vec<u8> = master[xover..].to_vec();
    let num_high = high.len() - 1;

    let mut low = Vec::with_capacity(num_high / 2 + 2);
    low.push(high[0]);
    // An odd band count leaves one band that both resolutions share, and it has to
    // be the first, so that pairing stays aligned to the top of the range.
    let mut at = if num_high % 2 == 1 { 1 } else { 0 };
    if at == 1 {
        low.push(high[1]);
    }
    while at + 2 <= num_high {
        at += 2;
        low.push(high[at]);
    }
    (low, high)
}

/// Place the noise floor bands over the low-resolution scalefactor bands.
fn noise_band_table(header: &SbrHeader, low: &[u8]) -> Result<Vec<u8>> {
    let num_low = low.len() - 1;
    let k_x = low[0] as usize;
    let k_high = *low.last().unwrap() as usize;

    let count = if header.noise_bands == 0 {
        1
    } else {
        ((header.noise_bands as f64 * log2(k_high, k_x)).round() as usize).max(1)
    };
    if count > MAX_NOISE_BANDS {
        return Err(bad("SBR header asks for more noise bands than the standard allows"));
    }

    let mut table = Vec::with_capacity(count + 1);
    table.push(low[0]);
    let mut at = 0usize;
    let mut remaining_low = num_low;
    let mut remaining_noise = count;
    for _ in 0..count {
        at += remaining_low / remaining_noise;
        table.push(low[at.min(num_low)]);
        remaining_low = num_low - at;
        remaining_noise -= 1;
    }
    Ok(table)
}

/// The nearest master band boundary to `goal`, searching upwards or downwards.
fn closest_master_edge(goal: usize, master: &[u8], upwards: bool) -> usize {
    let first = master[0] as usize;
    let last = *master.last().unwrap() as usize;
    if goal <= first {
        return first;
    }
    if goal >= last {
        return last;
    }
    if upwards {
        master.iter().map(|&v| v as usize).find(|&v| v >= goal).unwrap_or(last)
    } else {
        master.iter().rev().map(|&v| v as usize).find(|&v| v <= goal).unwrap_or(first)
    }
}

/// The subband each rate's patching aims to reach before wrapping.
fn patch_goal_band(output_rate_hz: u32) -> usize {
    match output_rate_hz {
        ..=32000 => 64,
        32001..=44100 => 46,
        44101..=48000 => 43,
        48001..=64000 => 32,
        64001..=88200 => 23,
        _ => 21,
    }
}

/// Work out which low subbands are copied where.
///
/// Each patch copies a run of the core's subbands upwards by an even stride, so
/// the copy preserves the odd/even alternation the QMF's modulation depends on.
/// Patches stack until the replicated range is filled.
fn build_patches(
    master: &[u8],
    k_x: usize,
    k_high: usize,
    output_rate_hz: u32,
) -> Result<Vec<Patch>> {
    let lsb = master[0] as usize;
    if lsb < FIRST_SOURCE_BAND + 4 {
        return Err(bad("SBR core range is too narrow to patch from"));
    }
    let xover_offset = k_x - lsb;

    let mut goal = closest_master_edge(patch_goal_band(output_rate_hz), master, true);
    if goal.abs_diff(k_high) < 4 {
        goal = k_high;
    }

    let mut source_floor = FIRST_SOURCE_BAND + xover_offset;
    let mut sb = lsb + xover_offset;
    if goal < sb && lsb > source_floor {
        return Err(bad("SBR patch target lies below the replicated range"));
    }

    let mut patches: Vec<Patch> = Vec::with_capacity(MAX_PATCHES);
    let mut previous_was_empty = false;

    while sb < k_high && patches.len() < MAX_PATCHES {
        let guard_start = sb;
        sb += GUARD_BANDS;
        let dst_start = sb;

        let mut width = goal as i32 - sb as i32;
        let available = lsb as i32 - source_floor as i32;
        let mut stop_after = false;
        if width <= 0 && width - available < 0 {
            stop_after = true;
        }
        if width - available >= 0 {
            // The patch would run past the core, so cap it where the source ends
            // and snap the end to a master band boundary.
            let stride = (sb - source_floor) & !1;
            let capped = lsb as i32 - (sb - stride) as i32;
            width = closest_master_edge((sb as i32 + capped) as usize, master, false) as i32
                - sb as i32;
        }

        let stride = ((width + sb as i32 - lsb as i32) + 1) & !1;
        if width > 0 {
            let src_start = (sb as i32 - stride).max(0) as usize;
            patches.push(Patch {
                dst_start,
                src_start,
                width: width as usize,
                guard_start,
            });
            sb += width as usize;
        }

        source_floor = FIRST_SOURCE_BAND;
        let near_goal = sb.abs_diff(goal) < 3;
        if near_goal {
            goal = k_high;
        } else if stop_after {
            break;
        }
        if width <= 0 {
            if previous_was_empty {
                break;
            }
            previous_was_empty = true;
        } else {
            previous_was_empty = false;
        }
    }

    // A final sliver of a patch would carry no useful signal; drop it.
    if patches.len() > 1 && patches[patches.len() - 1].width < 3 {
        patches.pop();
    }
    if patches.is_empty() {
        return Err(bad("SBR band layout yielded no patches"));
    }
    Ok(patches)
}

/// Bands per octave the limiter uses, as the reciprocal the merge test wants.
const LIMITER_BAND_RECIPROCALS: [f32; 4] = [0.25, 0.299_987_8, 0.5, 0.75];
/// Below this many octaves apart, two limiter borders are merged.
const LIMITER_MERGE_THRESHOLD: f32 = 502.0 / 4096.0;

/// Place the gain limiter's band borders.
///
/// The limiter wants bands roughly uniform in octaves, but it must also break at
/// every patch boundary, because a patch discontinuity is exactly where a runaway
/// gain would be audible. So the two sets of borders are merged and then thinned:
/// borders closer together than the target spacing are dropped, preferring to keep
/// the patch borders over the octave-spaced ones.
fn limiter_band_table(header: &SbrHeader, low: &[u8], patches: &[Patch]) -> Vec<u8> {
    let k_x = low[0] as usize;
    let k_high = *low.last().unwrap() as usize;
    let width = (k_high - k_x) as u8;

    if header.limiter_bands == 0 {
        return vec![0, width];
    }

    let mut patch_borders: Vec<u8> =
        patches.iter().map(|p| (p.guard_start - k_x) as u8).collect();
    patch_borders.push(width);

    let mut borders: Vec<u8> = low.iter().map(|&v| v - k_x as u8).collect();
    borders.extend(patch_borders[1..patch_borders.len() - 1].iter().copied());
    borders.sort_unstable();

    let reciprocal = LIMITER_BAND_RECIPROCALS[header.limiter_bands as usize];
    let mut kept: Vec<u8> = Vec::with_capacity(borders.len());
    kept.push(borders[0]);
    for &border in &borders[1..] {
        let previous = *kept.last().unwrap();
        if border == previous {
            continue;
        }
        let octaves =
            log2((border as usize) + k_x, (previous as usize) + k_x) as f32 * reciprocal;
        if octaves >= LIMITER_MERGE_THRESHOLD {
            kept.push(border);
            continue;
        }
        // Too close together: drop whichever of the two is not a patch border, and
        // if both are, keep them anyway.
        let border_is_patch = patch_borders.contains(&border);
        let previous_is_patch = patch_borders.contains(&previous);
        if !border_is_patch {
            continue;
        }
        if !previous_is_patch && kept.len() > 1 {
            kept.pop();
        }
        kept.push(border);
    }
    if *kept.last().unwrap() != width {
        kept.push(width);
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A default header at 44.1 kHz must produce a self-consistent layout.
    #[test]
    fn default_layout_is_consistent() {
        let header = SbrHeader::default();
        let layout = BandLayout::derive(&header, 44100).expect("layout");

        assert!(layout.master.windows(2).all(|w| w[0] < w[1]), "master table must increase");
        assert!(layout.low.windows(2).all(|w| w[0] < w[1]), "low table must increase");
        assert!(layout.high.windows(2).all(|w| w[0] < w[1]), "high table must increase");
        assert!(layout.noise.windows(2).all(|w| w[0] < w[1]), "noise table must increase");
        assert!(layout.limiter.windows(2).all(|w| w[0] < w[1]), "limiter table must increase");

        assert_eq!(layout.k_x, layout.low[0] as usize);
        assert_eq!(layout.k_high, *layout.low.last().unwrap() as usize);
        assert_eq!(layout.noise[0], layout.low[0]);
        assert_eq!(layout.noise.last(), layout.low.last());
        assert_eq!(layout.limiter[0], 0);
        assert_eq!(
            *layout.limiter.last().unwrap() as usize,
            layout.k_high - layout.k_x
        );
        assert!(layout.k_high <= SBR_SYNTHESIS_BANDS);
    }

    /// The low-resolution table must be a subset of the high-resolution one, and
    /// have about half as many bands.
    #[test]
    fn resolutions_nest() {
        for rate in [16000u32, 24000, 32000, 44100, 48000, 64000, 88200, 96000] {
            for start in 0..14u8 {
                let header = SbrHeader { start_freq: start, ..SbrHeader::default() };
                let Ok(layout) = BandLayout::derive(&header, rate) else { continue };
                for edge in &layout.low {
                    assert!(
                        layout.high.contains(edge),
                        "rate {rate} start {start}: low edge {edge} is not a high edge"
                    );
                }
                let (lo, hi) = (layout.sfb_count(false), layout.sfb_count(true));
                assert_eq!(lo, hi.div_ceil(2), "rate {rate} start {start}");
            }
        }
    }

    /// Every patch must read from inside the core range and write inside the
    /// replicated range, at an even stride.
    #[test]
    fn patches_stay_in_range() {
        for rate in [16000u32, 22050, 32000, 44100, 48000, 64000, 96000] {
            for start in 0..16u8 {
                for stop in [0u8, 3, 6, 9, 13] {
                    let header =
                        SbrHeader { start_freq: start, stop_freq: stop, ..SbrHeader::default() };
                    let Ok(layout) = BandLayout::derive(&header, rate) else { continue };
                    for patch in &layout.patches {
                        assert!(patch.width > 0);
                        assert!(
                            patch.src_start >= FIRST_SOURCE_BAND,
                            "rate {rate}: patch reads band {}",
                            patch.src_start
                        );
                        assert!(
                            patch.src_start + patch.width <= layout.k_x.max(layout.master[0] as usize),
                            "rate {rate}: patch reads past the core at {patch:?}, k_x {}",
                            layout.k_x
                        );
                        assert!(
                            patch.dst_start + patch.width <= layout.k_high,
                            "rate {rate}: patch writes past the range at {patch:?}"
                        );
                        assert_eq!(
                            (patch.dst_start - patch.src_start) % 2,
                            0,
                            "rate {rate}: patch stride is odd at {patch:?}"
                        );
                    }
                }
            }
        }
    }

    /// Deriving a layout must never panic, whatever the header says.
    #[test]
    fn every_header_is_survivable() {
        for rate in [16000u32, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000] {
            for start in 0..16u8 {
                for stop in 0..16u8 {
                    for scale in 0..4u8 {
                        for &alter in &[false, true] {
                            let header = SbrHeader {
                                start_freq: start,
                                stop_freq: stop,
                                freq_scale: scale,
                                alter_scale: alter,
                                ..SbrHeader::default()
                            };
                            let _ = BandLayout::derive(&header, rate);
                        }
                    }
                }
            }
        }
    }

    /// A header round-trips its fields through the bitstream.
    #[test]
    fn header_parses_its_fields() {
        use crate::bitstream::BitWriter;
        let mut w = BitWriter::new();
        w.write_bit(true); // amp_res: coarse
        w.write_bits(7, 4); // start_freq
        w.write_bits(3, 4); // stop_freq
        w.write_bits(5, 3); // xover_band
        w.write_bits(0, 2); // reserved
        w.write_bit(true); // extra 1
        w.write_bit(true); // extra 2
        w.write_bits(1, 2); // freq_scale
        w.write_bit(false); // alter_scale
        w.write_bits(3, 2); // noise_bands
        w.write_bits(1, 2); // limiter_bands
        w.write_bits(0, 2); // limiter_gains
        w.write_bit(false); // interpol_freq
        w.write_bit(false); // smoothing_mode
        w.write_bits(0, 8);
        let bytes = w.into_bytes();

        let mut r = BitReader::new(&bytes);
        let header = SbrHeader::parse(&mut r).unwrap();
        assert_eq!(header.amp_res, AmplitudeResolution::Coarse);
        assert_eq!(header.start_freq, 7);
        assert_eq!(header.stop_freq, 3);
        assert_eq!(header.xover_band, 5);
        assert_eq!(header.freq_scale, 1);
        assert!(!header.alter_scale);
        assert_eq!(header.noise_bands, 3);
        assert_eq!(header.limiter_bands, 1);
        assert_eq!(header.limiter_gains, 0);
        assert!(!header.interpol_freq);
        assert!(!header.smoothing_mode);
    }

    /// Omitting the extra header fields must restore their defaults rather than
    /// leave whatever the previous header set.
    #[test]
    fn absent_extras_take_defaults() {
        use crate::bitstream::BitWriter;
        let mut w = BitWriter::new();
        w.write_bit(false);
        w.write_bits(4, 4);
        w.write_bits(0, 4);
        w.write_bits(0, 3);
        w.write_bits(0, 2);
        w.write_bit(false);
        w.write_bit(false);
        w.write_bits(0, 8);
        let bytes = w.into_bytes();

        let mut r = BitReader::new(&bytes);
        let header = SbrHeader::parse(&mut r).unwrap();
        let defaults = SbrHeader::default();
        assert_eq!(header.freq_scale, defaults.freq_scale);
        assert_eq!(header.alter_scale, defaults.alter_scale);
        assert_eq!(header.noise_bands, defaults.noise_bands);
        assert_eq!(header.limiter_bands, defaults.limiter_bands);
        assert_eq!(header.limiter_gains, defaults.limiter_gains);
        assert_eq!(header.interpol_freq, defaults.interpol_freq);
        assert_eq!(header.smoothing_mode, defaults.smoothing_mode);
    }
}
