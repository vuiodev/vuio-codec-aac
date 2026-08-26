//! `individual_channel_stream()` parsing: window layout, sections, scalefactors,
//! pulses, TNS filters and spectral data.
//!
//! Follows ISO/IEC 14496-3 clause 4.4.2 and the reference C decoder
//! (`decoder/ixheaacd_channel.c`, `decoder/ixheaacd_block.c`).
//!
//! # Coefficient layout
//!
//! Spectral data is decoded in *grouped* order: for each window group, all bands of
//! all windows in that group are contiguous. [`deinterleave`] converts that into the
//! per-window layout the IMDCT consumes. Long-window frames have a single group of
//! one window, so the two layouts coincide.

use crate::bitstream::BitReader;
use crate::decoder::aac::huffman::{decode_scalefactor_delta, decode_spectral_band};
use crate::error::{DecodeError, Result};
use crate::tables::scalefactor::{MAX_SFB_LONG, MAX_WINDOWS, compute_sfb_offsets, get_sfb_table};
use crate::types::{AudioObjectType, FrameLength, SamplingRate, WindowSequence, WindowShape};

/// Codebook number reserved for bands with no coded spectrum.
pub const ZERO_HCB: u8 = 0;
/// Codebook number signalling perceptual noise substitution.
pub const NOISE_HCB: u8 = 13;
/// Codebook number signalling out-of-phase intensity stereo.
pub const INTENSITY_HCB2: u8 = 14;
/// Codebook number signalling in-phase intensity stereo.
pub const INTENSITY_HCB: u8 = 15;
/// Scalefactor bias: a scalefactor of 100 means unity gain.
pub const SF_OFFSET: i32 = 100;
/// Highest TNS filter order the standard permits for long windows.
pub const MAX_TNS_ORDER: usize = 20;
/// Maximum TNS filters per window.
pub const MAX_TNS_FILTERS: usize = 8;

/// Window layout and band structure for one channel's frame.
#[derive(Debug, Clone)]
pub struct IcsInfo {
    pub window_sequence: WindowSequence,
    pub window_shape: WindowShape,
    /// Highest coded scalefactor band, exclusive.
    pub max_sfb: usize,
    /// Number of window groups (1 for long windows, 1..=8 for short).
    pub num_window_groups: usize,
    /// Number of short windows in each group.
    pub window_group_length: [usize; MAX_WINDOWS],
    /// Total windows in the frame: 1 for long, 8 for `EIGHT_SHORT_SEQUENCE`.
    pub num_windows: usize,
    /// Scalefactor bands available at this rate and window length.
    pub num_swb: usize,
    /// Per-window band offsets; band `sfb` spans `swb_offset[sfb]..swb_offset[sfb+1]`.
    pub swb_offset: [u16; MAX_SFB_LONG + 1],
    /// Lines per window: frame length for long windows, one eighth for short.
    pub window_length: usize,
    /// Set when predictor/LTP data was signalled.
    pub predictor_data_present: bool,
}

impl Default for IcsInfo {
    fn default() -> Self {
        Self {
            window_sequence: WindowSequence::OnlyLongSequence,
            window_shape: WindowShape::Sine,
            max_sfb: 0,
            num_window_groups: 1,
            window_group_length: [1, 0, 0, 0, 0, 0, 0, 0],
            num_windows: 1,
            num_swb: 0,
            swb_offset: [0; MAX_SFB_LONG + 1],
            window_length: 1024,
            predictor_data_present: false,
        }
    }
}

impl IcsInfo {
    /// Parse `ics_info()` and derive the window-group and band layout.
    pub fn parse(
        reader: &mut BitReader,
        rate: SamplingRate,
        frame_length: FrameLength,
        aot: AudioObjectType,
        common_window: bool,
    ) -> Result<Self> {
        let _ics_reserved = reader.read_bit()?;
        let window_sequence = WindowSequence::from_u8(reader.read_u8(2)?)
            .unwrap_or(WindowSequence::OnlyLongSequence);
        let window_shape =
            WindowShape::from_u8(reader.read_u8(1)?).unwrap_or(WindowShape::Sine);

        let is_short = window_sequence.is_eight_short();
        let mut info = Self {
            window_sequence,
            window_shape,
            num_windows: if is_short { 8 } else { 1 },
            window_length: if is_short {
                frame_length.short_samples()
            } else {
                frame_length.samples()
            },
            ..Default::default()
        };

        if is_short {
            info.max_sfb = reader.read_u8(4)? as usize;
            let grouping = reader.read_u8(7)?;
            info.derive_groups(grouping);
        } else {
            info.max_sfb = reader.read_u8(6)? as usize;
            info.num_window_groups = 1;
            info.window_group_length = [1, 0, 0, 0, 0, 0, 0, 0];

            info.predictor_data_present = reader.read_bit()?;
            if info.predictor_data_present {
                Self::skip_predictor_data(reader, aot, common_window, info.max_sfb)?;
            }
        }

        // Band widths depend on the window length, so resolve them after the
        // window sequence is known.
        let widths = get_sfb_table(rate, is_short, frame_length);
        let mut offsets = [0usize; MAX_SFB_LONG + 1];
        let n = compute_sfb_offsets(widths, &mut offsets);
        info.num_swb = n - 1;
        for (dst, &src) in info.swb_offset.iter_mut().zip(offsets.iter()) {
            *dst = src as u16;
        }

        if info.max_sfb > info.num_swb {
            return Err(DecodeError::InvalidMaxSfb {
                max_sfb: info.max_sfb as u8,
                num_swb: info.num_swb as u8,
            }
            .into());
        }

        Ok(info)
    }

    /// Expand the 7-bit `scale_factor_grouping` field into group lengths.
    ///
    /// Each bit says whether short window `w + 1` continues the current group (1) or
    /// starts a new one (0), scanned from the most significant of the seven bits.
    fn derive_groups(&mut self, grouping: u8) {
        self.window_group_length = [0; MAX_WINDOWS];
        let mut g = 0usize;
        self.window_group_length[0] = 1;
        for w in 0..7 {
            if grouping & (1 << (6 - w)) != 0 {
                self.window_group_length[g] += 1;
            } else {
                g += 1;
                self.window_group_length[g] = 1;
            }
        }
        self.num_window_groups = g + 1;
    }

    /// Consume `predictor_data` / `ltp_data` without applying it.
    ///
    /// Main-profile backward prediction and LTP are not yet implemented; the fields
    /// still have to be consumed or every following field misaligns.
    fn skip_predictor_data(
        reader: &mut BitReader,
        aot: AudioObjectType,
        common_window: bool,
        max_sfb: usize,
    ) -> Result<()> {
        if aot == AudioObjectType::AacMain {
            // predictor_reset, then one bit per predictable band.
            if reader.read_bit()? {
                reader.skip_bits(5)?;
            }
            const PRED_SFB_MAX: usize = 41;
            reader.skip_bits(max_sfb.min(PRED_SFB_MAX))?;
        } else {
            // LTP for LC/LTP/scalable profiles, once per channel of the element.
            if reader.read_bit()? {
                skip_ltp_data(reader, max_sfb)?;
            }
            if common_window && reader.read_bit()? {
                skip_ltp_data(reader, max_sfb)?;
            }
        }
        Ok(())
    }

    /// Offsets of band `sfb` inside the grouped coefficient buffer for group `g`.
    ///
    /// Within a group the lines of one band from every window in that group sit
    /// together, so the band stride is the per-window width times the group length.
    #[inline]
    pub fn grouped_offset(&self, group: usize, sfb: usize) -> usize {
        let len = self.window_group_length[group];
        self.swb_offset[sfb] as usize * len
    }

    /// First coefficient index of window group `g` in the grouped buffer.
    #[inline]
    pub fn group_base(&self, group: usize) -> usize {
        let mut base = 0;
        for g in 0..group {
            base += self.window_group_length[g] * self.window_length;
        }
        base
    }

    /// Index of the first short window belonging to group `g`.
    #[inline]
    pub fn group_window_base(&self, group: usize) -> usize {
        self.window_group_length[..group].iter().sum()
    }
}

/// Consume one `ltp_data()` element.
fn skip_ltp_data(reader: &mut BitReader, max_sfb: usize) -> Result<()> {
    reader.skip_bits(11)?; // ltp_lag
    reader.skip_bits(3)?; // ltp_coef
    const MAX_LTP_SFB: usize = 40;
    reader.skip_bits(max_sfb.min(MAX_LTP_SFB))?;
    Ok(())
}

/// One TNS filter: a band range plus a PARCOR-coded all-pole shaping filter.
#[derive(Debug, Clone, Copy, Default)]
pub struct TnsFilterSpec {
    pub start_band: usize,
    pub stop_band: usize,
    pub order: usize,
    /// `true` when the filter runs from high to low frequency.
    pub downward: bool,
    /// Reflection-coefficient indices, already sign-extended.
    pub coef: [i8; MAX_TNS_ORDER],
    /// Coefficient resolution flag: 0 selects the 3-bit table, 1 the 4-bit table.
    pub resolution: u8,
}

/// TNS filters for every window of a frame.
#[derive(Debug, Clone)]
pub struct TnsData {
    pub present: bool,
    pub n_filt: [usize; MAX_WINDOWS],
    pub filters: [[TnsFilterSpec; MAX_TNS_FILTERS]; MAX_WINDOWS],
}

impl Default for TnsData {
    fn default() -> Self {
        Self {
            present: false,
            n_filt: [0; MAX_WINDOWS],
            filters: [[TnsFilterSpec::default(); MAX_TNS_FILTERS]; MAX_WINDOWS],
        }
    }
}

impl TnsData {
    /// Parse `tns_data()`.
    ///
    /// Filter band ranges are signalled as lengths counting down from the top band,
    /// matching `ixheaacd_channel.c`; this reconstructs absolute start/stop bands.
    pub fn parse(reader: &mut BitReader, ics: &IcsInfo) -> Result<Self> {
        let is_short = ics.window_sequence.is_eight_short();
        let (n_filt_bits, len_bits, order_bits) = if is_short { (1, 4, 3) } else { (2, 6, 5) };
        let max_order = if is_short { 7 } else { MAX_TNS_ORDER };

        let mut data = Self { present: true, ..Default::default() };

        for w in 0..ics.num_windows {
            let n_filt = reader.read_u8(n_filt_bits)? as usize;
            data.n_filt[w] = n_filt.min(MAX_TNS_FILTERS);
            if n_filt == 0 {
                continue;
            }
            let coef_res = reader.read_u8(1)?;

            let mut top = ics.num_swb;
            for f in 0..n_filt {
                let length = reader.read_u8(len_bits)? as usize;
                let start = top.saturating_sub(length);
                let order = reader.read_u8(order_bits)? as usize;
                if order > max_order {
                    return Err(DecodeError::InvalidTnsOrder(order as u8).into());
                }

                let mut filter = TnsFilterSpec {
                    start_band: start,
                    stop_band: top,
                    order,
                    resolution: coef_res,
                    ..Default::default()
                };
                top = start;

                if order > 0 {
                    filter.downward = reader.read_bit()?;
                    let coef_compress = reader.read_bit()? as usize;
                    // Compression drops the redundant high bit of the index.
                    let width = coef_res as usize + 3 - coef_compress;
                    let shift = 32 - width;
                    for i in 0..order {
                        let raw = reader.read_u32(width)? as i32;
                        // Sign-extend from `width` bits.
                        filter.coef[i] = ((raw << shift) >> shift) as i8;
                    }
                }

                if f < MAX_TNS_FILTERS {
                    data.filters[w][f] = filter;
                }
            }
        }
        Ok(data)
    }
}

/// A decoded `pulse_data()` element: up to four sparse amplitude corrections.
#[derive(Debug, Clone, Copy, Default)]
pub struct PulseData {
    pub number_pulse: usize,
    pub start_sfb: usize,
    pub offset: [u8; 4],
    pub amp: [u8; 4],
}

impl PulseData {
    /// Parse `pulse_data()`.
    pub fn parse(reader: &mut BitReader) -> Result<Self> {
        let number_pulse = reader.read_u8(2)? as usize + 1;
        let start_sfb = reader.read_u8(6)? as usize;
        let mut p = PulseData { number_pulse, start_sfb, ..Default::default() };
        for i in 0..number_pulse {
            p.offset[i] = reader.read_u8(5)?;
            p.amp[i] = reader.read_u8(4)?;
        }
        Ok(p)
    }

    /// Apply pulse corrections to the quantized long-window spectrum.
    ///
    /// Pulses are only legal on long windows, and offsets accumulate from the first
    /// line of `start_sfb`.
    pub fn apply(&self, ics: &IcsInfo, quant: &mut [i32]) {
        if self.start_sfb >= ics.num_swb {
            return;
        }
        let mut k = ics.swb_offset[self.start_sfb] as usize;
        for i in 0..self.number_pulse {
            k += self.offset[i] as usize;
            if k >= quant.len() {
                break;
            }
            // The pulse magnitude moves the coefficient away from zero.
            if quant[k] >= 0 {
                quant[k] += self.amp[i] as i32;
            } else {
                quant[k] -= self.amp[i] as i32;
            }
        }
    }
}

/// Everything decoded from one `individual_channel_stream()`.
#[derive(Debug, Clone)]
pub struct ChannelData {
    pub ics: IcsInfo,
    pub global_gain: u8,
    /// Codebook per `[group][band]`.
    pub sfb_cb: [[u8; MAX_SFB_LONG]; MAX_WINDOWS],
    /// Scalefactor (or intensity position / noise energy) per `[group][band]`.
    pub scale_factors: [[i16; MAX_SFB_LONG]; MAX_WINDOWS],
    pub tns: TnsData,
    pub pulse: Option<PulseData>,
    /// Quantized coefficients in grouped order.
    pub quant: Vec<i32>,
    /// Dequantized coefficients in per-window order.
    pub spec: Vec<f32>,
}

impl ChannelData {
    /// Allocate per-channel state for a frame of `frame_len` lines.
    pub fn new(frame_len: usize) -> Self {
        Self {
            ics: IcsInfo::default(),
            global_gain: 0,
            sfb_cb: [[0; MAX_SFB_LONG]; MAX_WINDOWS],
            scale_factors: [[0; MAX_SFB_LONG]; MAX_WINDOWS],
            tns: TnsData::default(),
            pulse: None,
            quant: vec![0; frame_len],
            spec: vec![0.0; frame_len],
        }
    }

    /// True when band `sfb` of group `g` carries intensity-stereo data.
    #[inline]
    pub fn is_intensity(&self, g: usize, sfb: usize) -> i32 {
        match self.sfb_cb[g][sfb] {
            INTENSITY_HCB => 1,
            INTENSITY_HCB2 => -1,
            _ => 0,
        }
    }

    /// True when band `sfb` of group `g` is noise-substituted.
    #[inline]
    pub fn is_noise(&self, g: usize, sfb: usize) -> bool {
        self.sfb_cb[g][sfb] == NOISE_HCB
    }
}

/// Parse one `individual_channel_stream()` into `out`.
///
/// `ics` is supplied when the element shares a window (a channel pair with
/// `common_window` set), in which case `ics_info()` is not present in this channel's
/// payload.
pub fn decode_ics(
    reader: &mut BitReader,
    out: &mut ChannelData,
    rate: SamplingRate,
    frame_length: FrameLength,
    aot: AudioObjectType,
    shared_ics: Option<&IcsInfo>,
) -> Result<()> {
    out.global_gain = reader.read_u8(8)?;

    out.ics = match shared_ics {
        Some(ics) => ics.clone(),
        None => IcsInfo::parse(reader, rate, frame_length, aot, false)?,
    };

    decode_section_data(reader, out)?;
    decode_scale_factor_data(reader, out)?;

    out.pulse = None;
    out.tns = TnsData::default();

    if reader.read_bit()? {
        let pulse = PulseData::parse(reader)?;
        // Pulses are undefined for short windows; ignore rather than misapply.
        if !out.ics.window_sequence.is_eight_short() {
            out.pulse = Some(pulse);
        }
    }
    if reader.read_bit()? {
        out.tns = TnsData::parse(reader, &out.ics)?;
    }
    if reader.read_bit()? {
        skip_gain_control_data(reader, &out.ics)?;
    }

    decode_spectral_data(reader, out)?;
    Ok(())
}

/// Parse `section_data()`, filling in the per-band codebook map.
fn decode_section_data(reader: &mut BitReader, out: &mut ChannelData) -> Result<()> {
    let is_short = out.ics.window_sequence.is_eight_short();
    let bits = if is_short { 3 } else { 5 };
    let escape = (1usize << bits) - 1;
    let max_sfb = out.ics.max_sfb;

    for g in 0..out.ics.num_window_groups {
        out.sfb_cb[g] = [ZERO_HCB; MAX_SFB_LONG];
        let mut k = 0usize;
        while k < max_sfb {
            let cb = reader.read_u8(4)?;
            let mut len = 0usize;
            loop {
                let incr = reader.read_u8(bits)? as usize;
                len += incr;
                if incr != escape {
                    break;
                }
                if k + len > max_sfb {
                    break;
                }
            }
            let end = (k + len).min(max_sfb);
            if end <= k {
                // A zero-length section would loop forever on a corrupt stream.
                return Err(DecodeError::InvalidSectionLength.into());
            }
            for sfb in k..end {
                out.sfb_cb[g][sfb] = cb;
            }
            k = end;
        }
    }
    Ok(())
}

/// Parse `scale_factor_data()`, DPCM-decoding scalefactors, intensity positions and
/// noise energies from their shared Huffman codebook.
fn decode_scale_factor_data(reader: &mut BitReader, out: &mut ChannelData) -> Result<()> {
    let mut scale_factor = out.global_gain as i32;
    let mut is_position = 0i32;
    // Noise energy starts biased down from the global gain by 90 (ISO 4.6.2).
    let mut noise_energy = out.global_gain as i32 - 90;
    let mut noise_pcm_flag = true;

    for g in 0..out.ics.num_window_groups {
        for sfb in 0..out.ics.max_sfb {
            let cb = out.sfb_cb[g][sfb];
            if cb == ZERO_HCB {
                out.scale_factors[g][sfb] = 0;
                continue;
            }
            if cb == INTENSITY_HCB || cb == INTENSITY_HCB2 {
                is_position += decode_scalefactor_delta(reader)?;
                out.scale_factors[g][sfb] = is_position as i16;
            } else if cb == NOISE_HCB {
                if noise_pcm_flag {
                    noise_pcm_flag = false;
                    noise_energy += reader.read_u32(9)? as i32 - 256;
                } else {
                    noise_energy += decode_scalefactor_delta(reader)?;
                }
                out.scale_factors[g][sfb] = noise_energy as i16;
            } else {
                scale_factor += decode_scalefactor_delta(reader)?;
                if !(0..=255).contains(&scale_factor) {
                    return Err(DecodeError::ScalefactorOutOfRange(scale_factor).into());
                }
                out.scale_factors[g][sfb] = scale_factor as i16;
            }
        }
    }
    Ok(())
}

/// Consume `gain_control_data()`.
///
/// Only used by the SSR profile, which this decoder does not implement; the fields
/// still have to be skipped to stay bit-aligned.
fn skip_gain_control_data(reader: &mut BitReader, ics: &IcsInfo) -> Result<()> {
    let max_band = reader.read_u8(2)? as usize;
    let windows = match ics.window_sequence {
        WindowSequence::OnlyLongSequence => 1,
        WindowSequence::LongStartSequence | WindowSequence::LongStopSequence => 2,
        WindowSequence::EightShortSequence => 8,
    };
    let len_bits = if ics.window_sequence == WindowSequence::EightShortSequence { 3 } else { 5 };
    for _ in 0..max_band {
        for w in 0..windows {
            let adjust_num = reader.read_u8(3)? as usize;
            for _ in 0..adjust_num {
                reader.skip_bits(4)?; // alevcode
                reader.skip_bits(if w == 0 { 5 } else { len_bits })?; // aloccode
            }
        }
    }
    Ok(())
}

/// Parse `spectral_data()` into the grouped quantized buffer.
fn decode_spectral_data(reader: &mut BitReader, out: &mut ChannelData) -> Result<()> {
    out.quant.fill(0);
    let ics = out.ics.clone();

    for g in 0..ics.num_window_groups {
        let group_base = ics.group_base(g);
        let group_len = ics.window_group_length[g];
        for sfb in 0..ics.max_sfb {
            let cb = out.sfb_cb[g][sfb];
            // ZERO, NOISE and INTENSITY bands carry no Huffman-coded spectrum.
            if cb == ZERO_HCB || cb == NOISE_HCB || cb == INTENSITY_HCB || cb == INTENSITY_HCB2 {
                continue;
            }
            let start = group_base + ics.grouped_offset(g, sfb);
            let width =
                (ics.swb_offset[sfb + 1] - ics.swb_offset[sfb]) as usize * group_len;
            let end = (start + width).min(out.quant.len());
            if start >= end {
                continue;
            }
            decode_spectral_band(reader, cb, &mut out.quant[start..end])?;
        }
    }

    if let Some(pulse) = out.pulse {
        pulse.apply(&ics, &mut out.quant);
    }
    Ok(())
}

/// Rearrange grouped coefficients into per-window order.
///
/// Input holds, for each group, every band of every window in that group
/// contiguously. Output holds `num_windows` consecutive windows of
/// `window_length` lines each, which is what the IMDCT expects.
pub fn deinterleave(ics: &IcsInfo, grouped: &[f32], out: &mut [f32]) {
    if !ics.window_sequence.is_eight_short() {
        out[..grouped.len()].copy_from_slice(grouped);
        return;
    }

    out.fill(0.0);
    let win_len = ics.window_length;
    for g in 0..ics.num_window_groups {
        let group_base = ics.group_base(g);
        let group_len = ics.window_group_length[g];
        let win_base = ics.group_window_base(g);
        for sfb in 0..ics.num_swb {
            let lo = ics.swb_offset[sfb] as usize;
            let hi = ics.swb_offset[sfb + 1] as usize;
            let width = hi - lo;
            for w in 0..group_len {
                let src = group_base + lo * group_len + w * width;
                let dst = (win_base + w) * win_len + lo;
                if src + width <= grouped.len() && dst + width <= out.len() {
                    out[dst..dst + width].copy_from_slice(&grouped[src..src + width]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `scale_factor_grouping` expansion must always describe eight short windows.
    #[test]
    fn window_grouping_covers_eight_windows() {
        for grouping in 0..128u8 {
            let mut info = IcsInfo {
                window_sequence: WindowSequence::EightShortSequence,
                ..Default::default()
            };
            info.derive_groups(grouping);
            let total: usize = info.window_group_length[..info.num_window_groups].iter().sum();
            assert_eq!(total, 8, "grouping {grouping:#09b} covered {total} windows");
            assert!((1..=8).contains(&info.num_window_groups));
        }
    }

    /// All ones means one group of eight; all zeros means eight groups of one.
    #[test]
    fn grouping_extremes() {
        let mut all_ones = IcsInfo {
            window_sequence: WindowSequence::EightShortSequence,
            ..Default::default()
        };
        all_ones.derive_groups(0b111_1111);
        assert_eq!(all_ones.num_window_groups, 1);
        assert_eq!(all_ones.window_group_length[0], 8);

        let mut all_zeros = IcsInfo {
            window_sequence: WindowSequence::EightShortSequence,
            ..Default::default()
        };
        all_zeros.derive_groups(0);
        assert_eq!(all_zeros.num_window_groups, 8);
        assert!(all_zeros.window_group_length[..8].iter().all(|&l| l == 1));
    }

    /// Deinterleaving grouped short-window data must land every line in the window
    /// and band the encoder put it in.
    #[test]
    fn deinterleave_round_trips_short_windows() {
        let mut ics = IcsInfo {
            window_sequence: WindowSequence::EightShortSequence,
            num_windows: 8,
            window_length: 128,
            ..Default::default()
        };
        // Two groups of two and one of four.
        ics.derive_groups(0b100_0100);
        let widths = crate::tables::sfb::SFB_48_128;
        let mut offsets = [0usize; MAX_SFB_LONG + 1];
        let n = compute_sfb_offsets(widths, &mut offsets);
        ics.num_swb = n - 1;
        ics.max_sfb = ics.num_swb;
        for (d, &s) in ics.swb_offset.iter_mut().zip(offsets.iter()) {
            *d = s as u16;
        }

        // Tag every line with its (window, line) identity through the grouped layout.
        let mut grouped = vec![0.0f32; 1024];
        for g in 0..ics.num_window_groups {
            let base = ics.group_base(g);
            let glen = ics.window_group_length[g];
            let wbase = ics.group_window_base(g);
            for sfb in 0..ics.num_swb {
                let lo = ics.swb_offset[sfb] as usize;
                let hi = ics.swb_offset[sfb + 1] as usize;
                for w in 0..glen {
                    for k in lo..hi {
                        let src = base + lo * glen + w * (hi - lo) + (k - lo);
                        grouped[src] = ((wbase + w) * 128 + k) as f32;
                    }
                }
            }
        }

        let mut out = vec![-1.0f32; 1024];
        deinterleave(&ics, &grouped, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, i as f32, "line {i} landed wrong");
        }
    }

    /// Long windows pass through deinterleaving unchanged.
    #[test]
    fn deinterleave_is_identity_for_long_windows() {
        let ics = IcsInfo { window_length: 1024, ..Default::default() };
        let grouped: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 1024];
        deinterleave(&ics, &grouped, &mut out);
        assert_eq!(out, grouped);
    }

    /// Group bases must partition the frame without gaps or overlap.
    #[test]
    fn group_bases_partition_the_frame() {
        for grouping in 0..128u8 {
            let mut ics = IcsInfo {
                window_sequence: WindowSequence::EightShortSequence,
                num_windows: 8,
                window_length: 128,
                ..Default::default()
            };
            ics.derive_groups(grouping);
            let mut expected = 0;
            for g in 0..ics.num_window_groups {
                assert_eq!(ics.group_base(g), expected);
                expected += ics.window_group_length[g] * 128;
            }
            assert_eq!(expected, 1024);
        }
    }
}
