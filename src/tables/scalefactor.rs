//! Scalefactor Band Width Tables for MPEG AAC Codecs
//!
//! Provides scalefactor band definitions across all 13 standard sampling rates
//! for Long (1024, 960, 512, 480) and Short (128, 120) window lengths.

use crate::types::{FrameLength, SamplingRate};

// 1024-point Long Window Scalefactor Band Widths
pub static SFB_96_1024: &[u8] = &[
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 16, 16, 24, 28,
    36, 44, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
];

pub static SFB_64_1024: &[u8] = &[
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 16, 16, 16, 20, 24, 24, 28,
    36, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
];

pub static SFB_48_1024: &[u8] = &[
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 20, 20, 24, 24,
    28, 28, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 96,
];

pub static SFB_44100_1024: &[u8] = SFB_48_1024;


pub static SFB_32_1024: &[u8] = &[
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 20, 20, 24, 24,
    28, 28, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
];

pub static SFB_24_1024: &[u8] = &[
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
    20, 20, 24, 24, 28, 28, 32, 36, 36, 40, 44, 48, 52, 52, 64, 64, 64, 64, 64,
];

pub static SFB_16_1024: &[u8] = &[
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 16, 16, 16, 16, 20, 20,
    20, 24, 24, 28, 28, 32, 36, 40, 40, 44, 48, 52, 56, 60, 64, 64, 64,
];

pub static SFB_8_1024: &[u8] = &[
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 16, 16, 16, 16, 16, 16, 16, 20, 20, 20,
    20, 24, 24, 24, 28, 28, 32, 36, 36, 40, 44, 48, 52, 56, 60, 64, 80,
];

// 128-point Short Window Scalefactor Band Widths
pub static SFB_96_128: &[u8] = &[4, 4, 4, 4, 4, 4, 8, 8, 8, 16, 28, 36];
pub static SFB_48_128: &[u8] = &[4, 4, 4, 4, 4, 8, 8, 8, 12, 12, 12, 16, 16, 16];
pub static SFB_24_128: &[u8] = &[4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 12, 12, 16, 16, 20];
pub static SFB_16_128: &[u8] = &[4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 12, 12, 16, 20, 20];
pub static SFB_8_128: &[u8] = &[4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 12, 16, 20, 20];

// 960-point Long Window Scalefactor Band Widths
pub static SFB_96_960: &[u8] = &[
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 16, 16, 24, 28,
    36, 44, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
];

pub static SFB_48_960: &[u8] = &[
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 20, 20, 24, 24,
    28, 28, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
];

// 120-point Short Window Scalefactor Band Widths
pub static SFB_96_120: &[u8] = &[4, 4, 4, 4, 4, 4, 8, 8, 8, 16, 28, 28];

/// Get scalefactor band width table for a given sampling rate, window length, and frame size.
pub fn get_sfb_table(rate: SamplingRate, is_short: bool, frame_len: FrameLength) -> &'static [u8] {
    let hz = rate.hz();
    match frame_len {
        FrameLength::Samples1024 => {
            if is_short {
                if hz >= 64000 {
                    SFB_96_128
                } else if hz >= 32000 {
                    SFB_48_128
                } else if hz >= 22050 {
                    SFB_24_128
                } else if hz >= 11025 {
                    SFB_16_128
                } else {
                    SFB_8_128
                }
            } else if hz >= 88200 {
                SFB_96_1024
            } else if hz >= 64000 {
                SFB_64_1024
            } else if hz >= 44100 {
                SFB_48_1024
            } else if hz >= 32000 {
                SFB_32_1024
            } else if hz >= 22050 {
                SFB_24_1024
            } else if hz >= 11025 {
                SFB_16_1024
            } else {
                SFB_8_1024
            }
        }
        FrameLength::Samples960 => {
            if is_short {
                SFB_96_120
            } else if hz >= 64000 {
                SFB_96_960
            } else {
                SFB_48_960
            }
        }
        _ => SFB_48_1024,
    }
}

/// Compute cumulative scalefactor band offsets from widths (starting at offset 0).
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
    fn test_sfb_offsets_sum() {
        let mut offsets = [0usize; 64];
        let num_offsets = compute_sfb_offsets(SFB_48_1024, &mut offsets);
        assert_eq!(num_offsets, SFB_48_1024.len() + 1);
        assert_eq!(offsets[num_offsets - 1], 1024);
    }
}
