//! Scalefactor-band layout selection.
//!
//! The band-width tables themselves live in [`crate::tables::sfb`], generated
//! directly from the reference C ROM. This module maps a (sampling rate, window
//! length, frame length) triple onto the right table and turns widths into the
//! cumulative offsets the decoder indexes with.

use crate::tables::sfb;
use crate::types::{FrameLength, SamplingRate};

pub use crate::tables::sfb::*;

/// Alias kept for call sites that name the 44.1 kHz long table directly; 44.1 kHz
/// and 48 kHz share one band layout in AAC.
pub static SFB_44100_1024: &[u8] = sfb::SFB_48_1024;

/// Largest number of scalefactor bands in any long-window table (32 kHz, 51 bands).
pub const MAX_SFB_LONG: usize = 51;

/// Number of short windows in an `EIGHT_SHORT_SEQUENCE` frame.
pub const MAX_WINDOWS: usize = 8;

/// Lower bounds that snap an arbitrary rate onto a standard sampling-rate index.
///
/// Copied from `ixheaacd_sampling_boundaries` so that non-standard rates (legal in
/// an `AudioSpecificConfig`) pick the same band layout the reference decoder does.
const SAMPLING_BOUNDARIES: [u32; 12] = [
    92017, 75132, 55426, 46009, 37566, 27713, 23004, 18783, 13856, 11502, 9391, 0,
];

/// Resolve any rate in Hz to a standard sampling-rate index (0..=11).
pub const fn sampling_rate_index(hz: u32) -> usize {
    let mut i = 0;
    while i < SAMPLING_BOUNDARIES.len() {
        if SAMPLING_BOUNDARIES[i] <= hz {
            return i;
        }
        i += 1;
    }
    11
}

/// Band-layout group for a sampling-rate index.
///
/// The 13 standard rates share seven long-window layouts: 96/88.2, 64, 48/44.1, 32,
/// 24/22.05, 16/12/11.025, and 8/7.35 kHz.
const RATE_INDEX_TO_GROUP: [usize; 13] = [0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 5, 6, 6];

/// Group index of the sampling-rate-dependent band tables.
const fn layout_group(hz: u32) -> usize {
    RATE_INDEX_TO_GROUP[sampling_rate_index(hz)]
}

/// Long-window band tables by layout group, for 1024-line frames.
static LONG_1024: [&[u8]; 7] = [
    sfb::SFB_96_1024,
    sfb::SFB_64_1024,
    sfb::SFB_48_1024,
    sfb::SFB_32_1024,
    sfb::SFB_24_1024,
    sfb::SFB_16_1024,
    sfb::SFB_8_1024,
];

/// Short-window band tables by layout group, for 128-line windows.
///
/// The 64 kHz group reuses the 96 kHz short table, and 32 kHz reuses 48 kHz, which
/// is why this differs from [`LONG_1024`] in more than scale.
static SHORT_128: [&[u8]; 7] = [
    sfb::SFB_96_128,
    sfb::SFB_96_128,
    sfb::SFB_48_128,
    sfb::SFB_48_128,
    sfb::SFB_24_128,
    sfb::SFB_16_128,
    sfb::SFB_8_128,
];

/// Long-window band tables by layout group, for 960-line frames.
static LONG_960: [&[u8]; 7] = [
    sfb::SFB_96_960,
    sfb::SFB_64_960,
    sfb::SFB_48_960,
    sfb::SFB_48_960,
    sfb::SFB_24_960,
    sfb::SFB_16_960,
    sfb::SFB_8_960,
];

/// Short-window band tables by layout group, for 120-line windows.
static SHORT_120: [&[u8]; 7] = [
    sfb::SFB_96_120,
    sfb::SFB_96_120,
    sfb::SFB_48_120,
    sfb::SFB_48_120,
    sfb::SFB_24_120,
    sfb::SFB_16_120,
    sfb::SFB_8_120,
];

/// Long-window band tables for 512-line (AAC-LD) frames.
///
/// Only 48/44.1, 32 and 24/22.05 kHz are defined for low-delay operation.
static LONG_512: [&[u8]; 7] = [
    sfb::SFB_48_512,
    sfb::SFB_48_512,
    sfb::SFB_48_512,
    sfb::SFB_32_512,
    sfb::SFB_24_512,
    sfb::SFB_24_512,
    sfb::SFB_24_512,
];

/// Long-window band tables for 480-line (AAC-LD) frames.
static LONG_480: [&[u8]; 7] = [
    sfb::SFB_48_480,
    sfb::SFB_48_480,
    sfb::SFB_48_480,
    sfb::SFB_32_480,
    sfb::SFB_24_480,
    sfb::SFB_24_480,
    sfb::SFB_24_480,
];

/// Get the scalefactor band width table for a sampling rate and window length.
pub fn get_sfb_table(rate: SamplingRate, is_short: bool, frame_len: FrameLength) -> &'static [u8] {
    let g = layout_group(rate.hz());
    match frame_len {
        FrameLength::Samples1024 | FrameLength::Samples768 => {
            if is_short { SHORT_128[g] } else { LONG_1024[g] }
        }
        FrameLength::Samples960 => {
            if is_short { SHORT_120[g] } else { LONG_960[g] }
        }
        // Low-delay frames have no short-window mode; the long table applies.
        FrameLength::Samples512 => LONG_512[g],
        FrameLength::Samples480 => LONG_480[g],
    }
}

/// Convert band widths into cumulative offsets, returning the number of offsets.
///
/// `offsets[0]` is always 0 and `offsets[n]` is the total line count, so the band
/// `sfb` spans `offsets[sfb]..offsets[sfb + 1]`.
pub fn compute_sfb_offsets(widths: &[u8], offsets: &mut [usize]) -> usize {
    offsets[0] = 0;
    let mut sum = 0;
    for (i, &w) in widths.iter().enumerate() {
        sum += w as usize;
        offsets[i + 1] = sum;
    }
    widths.len() + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sfb_offsets_sum_to_frame_length() {
        let mut offsets = [0usize; 64];
        let n = compute_sfb_offsets(SFB_48_1024, &mut offsets);
        assert_eq!(n, SFB_48_1024.len() + 1);
        assert_eq!(offsets[n - 1], 1024);
    }

    /// Every long table must tile a full frame and every short table a full window.
    #[test]
    fn all_tables_tile_their_transform() {
        for &rate in &[
            SamplingRate::Hz96000,
            SamplingRate::Hz88200,
            SamplingRate::Hz64000,
            SamplingRate::Hz48000,
            SamplingRate::Hz44100,
            SamplingRate::Hz32000,
            SamplingRate::Hz24000,
            SamplingRate::Hz22050,
            SamplingRate::Hz16000,
            SamplingRate::Hz12000,
            SamplingRate::Hz11025,
            SamplingRate::Hz8000,
            SamplingRate::Hz7350,
        ] {
            for (frame, long_total, short_total) in [
                (FrameLength::Samples1024, 1024usize, 128usize),
                (FrameLength::Samples960, 960, 120),
            ] {
                let long: usize = get_sfb_table(rate, false, frame).iter().map(|&w| w as usize).sum();
                assert_eq!(long, long_total, "{rate:?} long table for {frame:?}");
                let short: usize = get_sfb_table(rate, true, frame).iter().map(|&w| w as usize).sum();
                assert_eq!(short, short_total, "{rate:?} short table for {frame:?}");
            }
        }
    }

    /// The band count must never exceed the buffer bound the decoder relies on.
    #[test]
    fn band_counts_fit_max_sfb_long() {
        for &rate in &[SamplingRate::Hz96000, SamplingRate::Hz32000, SamplingRate::Hz8000] {
            let n = get_sfb_table(rate, false, FrameLength::Samples1024).len();
            assert!(n <= MAX_SFB_LONG, "{rate:?} has {n} bands, over {MAX_SFB_LONG}");
        }
    }

    /// 44.1 kHz and 48 kHz share a layout; 32 kHz has its own, wider one.
    #[test]
    fn rate_groups_match_the_standard() {
        let t48 = get_sfb_table(SamplingRate::Hz48000, false, FrameLength::Samples1024);
        let t441 = get_sfb_table(SamplingRate::Hz44100, false, FrameLength::Samples1024);
        let t32 = get_sfb_table(SamplingRate::Hz32000, false, FrameLength::Samples1024);
        assert_eq!(t48, t441);
        assert_eq!(t48.len(), 49);
        assert_eq!(t32.len(), 51);
    }
}
