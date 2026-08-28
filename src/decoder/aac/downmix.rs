//! Fixed multichannel-to-stereo downmix, for output devices that cannot
//! render more than two channels (`ixheaacd_dec_downmix_to_stereo`,
//! ported from `c/libxaac/decoder/ixheaacd_multichannel.c`).
//!
//! # The matrix, and the channel order it assumes
//!
//! This is a *fixed* downmix — four hardcoded 2-row matrices, one per source
//! layout, not the transmitted `matrix_mixdown_idx` a `program_config_element()`
//! can carry (that is a different, per-stream mechanism this module does not
//! implement). Reverse-engineered from which matrix columns are shared between
//! the two output rows in `ixheaacd_common_rom.c`'s `down_mix_martix[4][2][8]`
//! table (Q30 fixed-point; the values here are that table divided by 2^30), the
//! channel each column expects is:
//!
//! | Layout | Column order |
//! |---|---|
//! | [`Layout::Ch5_0`]  | L, R, C, Ls, Rs |
//! | [`Layout::Ch5_1`]  | L, R, C, LFE, Ls, Rs |
//! | [`Layout::Ch7_0`]  | L, R, C, Ls, Rs, Lrs, Rrs |
//! | [`Layout::Ch7_1`]  | L, R, ?, ?, Ls, Rs, Lrs, Rrs |
//!
//! In every layout but 7.1, a column feeding both output rows (center, and
//! LFE where present) is mixed in at a reduced, *equal* level, and every
//! other column pans hard to one side with zero leakage into the other —
//! confirmed directly against the table's numbers, not assumed. 7.1's
//! columns 2 and 3 break that pattern: each feeds both rows, but at two
//! *different* weights that mirror between the two columns (`0.211`/`0.070`
//! and `0.070`/`0.211` — the same `0.211` a hard-panned surround column gets,
//! plus a partial `0.070` bleed to the opposite side). That is consistent
//! with a pair of wide front channels rather than a plain center/LFE pair,
//! but nothing in the reference documents column semantics explicitly, so
//! this module does not assert channel names for those two columns and
//! [`downmix_to_stereo`] is agnostic to what any column represents — it only
//! needs the matrix and the caller's channel order to agree.
//!
//! # What is not covered
//!
//! The reference's caller resolves each channel to its matrix column via
//! `aac_config.slot_element[]`, built from the stream's actual element order
//! (SCE/CPE/LFE) and PCE-declared channel roles; nothing in this crate tracks
//! that mapping today (see `text/plan.txt` phase 7.6), so this function takes
//! channels already in the canonical order the table above lists rather than
//! resolving it from a decoded frame's element list itself.

/// Which fixed downmix matrix applies, named by the source layout's
/// channel count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// 5 channels, no LFE: L, R, C, Ls, Rs.
    Ch5_0,
    /// 5.1: L, R, C, LFE, Ls, Rs.
    Ch5_1,
    /// 7 channels, no LFE: L, R, C, Ls, Rs, Lrs, Rrs.
    Ch7_0,
    /// 7.1: L, R, C, LFE, Ls, Rs, Lrs, Rrs.
    Ch7_1,
}

impl Layout {
    /// How many input channels this layout expects.
    pub const fn channels(self) -> usize {
        match self {
            Layout::Ch5_0 => 5,
            Layout::Ch5_1 => 6,
            Layout::Ch7_0 => 7,
            Layout::Ch7_1 => 8,
        }
    }

    fn matrix(self) -> &'static [[f32; 8]; 2] {
        &DOWNMIX_MATRIX[match self {
            Layout::Ch5_0 => 0,
            Layout::Ch5_1 => 1,
            Layout::Ch7_0 => 2,
            Layout::Ch7_1 => 3,
        }]
    }
}

/// `down_mix_martix[4][2][8]`, Q30 values divided by 2^30.
#[rustfmt::skip]
const DOWNMIX_MATRIX: [[[f32; 8]; 2]; 4] = [
    // 5.0: L, R, C, Ls, Rs
    [
        [0.377358491, 0.0, 0.293405304, 0.293405304, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.377358491, 0.293405304, 0.0, 0.293405304, 0.0, 0.0, 0.0],
    ],
    // 5.1: L, R, C, LFE, Ls, Rs
    [
        [0.377358491, 0.0, 0.266832747, 0.088944249, 0.266832747, 0.0, 0.0, 0.0],
        [0.0, 0.377358491, 0.266832747, 0.088944249, 0.0, 0.266832747, 0.0, 0.0],
    ],
    // 7.0: L, R, C, Ls, Rs, Lrs, Rrs
    [
        [0.377358491, 0.0, 0.227365525, 0.227365525, 0.0, 0.227365525, 0.0, 0.0],
        [0.0, 0.377358491, 0.227365525, 0.0, 0.227365525, 0.0, 0.227365525, 0.0],
    ],
    // 7.1: L, R, C, LFE, Ls, Rs, Lrs, Rrs
    [
        [0.377358491, 0.0, 0.211076651, 0.070358884, 0.211076651, 0.0, 0.211076651, 0.0],
        [0.0, 0.377358491, 0.070358884, 0.211076651, 0.0, 0.211076651, 0.0, 0.211076651],
    ],
];

/// Downmix `channels` (one slice per input channel, in the column order
/// [`Layout`] documents, all the same length) to a stereo pair.
///
/// # Panics
///
/// If `channels.len()` does not match `layout.channels()`, or the channels
/// are not all the same length.
pub fn downmix_to_stereo(layout: Layout, channels: &[&[f32]]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(channels.len(), layout.channels(), "channel count must match the layout");
    let n = channels.first().map_or(0, |c| c.len());
    assert!(channels.iter().all(|c| c.len() == n), "all channels must be the same length");

    let matrix = layout.matrix();
    let mut left = vec![0.0f32; n];
    let mut right = vec![0.0f32; n];
    for (col, channel) in channels.iter().enumerate() {
        let (gl, gr) = (matrix[0][col], matrix[1][col]);
        if gl == 0.0 && gr == 0.0 {
            continue;
        }
        for i in 0..n {
            left[i] += gl * channel[i];
            right[i] += gr * channel[i];
        }
    }
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn impulse_channels(layout: Layout, active: usize, n: usize) -> Vec<Vec<f32>> {
        (0..layout.channels())
            .map(|c| {
                let mut v = vec![0.0f32; n];
                if c == active {
                    v[0] = 1.0;
                }
                v
            })
            .collect()
    }

    /// The center channel (and LFE, where present) must land equally in both
    /// outputs -- that is the whole point of a "center" channel. Confirmed
    /// for every layout except 7.1, whose columns 2/3 are not a plain
    /// center/LFE pair (see this module's docs) and are covered separately.
    #[test]
    fn center_and_lfe_split_evenly_between_left_and_right() {
        for (layout, center_col) in [(Layout::Ch5_0, 2), (Layout::Ch5_1, 2), (Layout::Ch7_0, 2)] {
            let chans = impulse_channels(layout, center_col, 1);
            let refs: Vec<&[f32]> = chans.iter().map(|v| v.as_slice()).collect();
            let (l, r) = downmix_to_stereo(layout, &refs);
            assert_eq!(l[0], r[0], "{layout:?}: center must split evenly, got L={} R={}", l[0], r[0]);
            assert!(l[0] > 0.0);
        }
        // LFE, present only in 5.1 (7.1's equivalent column is not a plain LFE -- see below).
        let chans = impulse_channels(Layout::Ch5_1, 3, 1);
        let refs: Vec<&[f32]> = chans.iter().map(|v| v.as_slice()).collect();
        let (l, r) = downmix_to_stereo(Layout::Ch5_1, &refs);
        assert_eq!(l[0], r[0], "5.1: LFE must split evenly, got L={} R={}", l[0], r[0]);
        assert!(l[0] > 0.0);
    }

    /// 7.1's columns 2 and 3 feed both outputs but at two different weights
    /// that mirror between the columns, not an even split -- verified
    /// directly against the reference's own Q30 values rather than assumed.
    #[test]
    fn ch7_1_columns_2_and_3_are_asymmetric_mirrored_pairs() {
        let chans = impulse_channels(Layout::Ch7_1, 2, 1);
        let refs: Vec<&[f32]> = chans.iter().map(|v| v.as_slice()).collect();
        let (l, r) = downmix_to_stereo(Layout::Ch7_1, &refs);
        assert!((l[0] - 0.211_076_65).abs() < 1e-5);
        assert!((r[0] - 0.070_358_88).abs() < 1e-5);

        let chans = impulse_channels(Layout::Ch7_1, 3, 1);
        let refs: Vec<&[f32]> = chans.iter().map(|v| v.as_slice()).collect();
        let (l, r) = downmix_to_stereo(Layout::Ch7_1, &refs);
        assert!((l[0] - 0.070_358_88).abs() < 1e-5);
        assert!((r[0] - 0.211_076_65).abs() < 1e-5);
    }

    /// Every surround channel must pan hard to its own side: zero leakage
    /// into the opposite output, matching the zeros in the reference table.
    #[test]
    fn surround_channels_pan_hard_with_no_crosstalk() {
        // (layout, column, which side must be nonzero)
        let cases = [
            (Layout::Ch5_0, 3, true),  // Ls -> left only
            (Layout::Ch5_0, 4, false), // Rs -> right only
            (Layout::Ch7_1, 4, true),  // Ls
            (Layout::Ch7_1, 5, false), // Rs
            (Layout::Ch7_1, 6, true),  // Lrs
            (Layout::Ch7_1, 7, false), // Rrs
        ];
        for (layout, col, left_side) in cases {
            let chans = impulse_channels(layout, col, 1);
            let refs: Vec<&[f32]> = chans.iter().map(|v| v.as_slice()).collect();
            let (l, r) = downmix_to_stereo(layout, &refs);
            if left_side {
                assert!(l[0] > 0.0 && r[0] == 0.0, "{layout:?} col {col}: expected left-only, got L={} R={}", l[0], r[0]);
            } else {
                assert!(r[0] > 0.0 && l[0] == 0.0, "{layout:?} col {col}: expected right-only, got L={} R={}", l[0], r[0]);
            }
        }
    }

    /// L must map straight to the left output at the front-channel level and
    /// not leak into the right output, and symmetrically for R.
    #[test]
    fn front_left_and_right_map_straight_through() {
        let chans = impulse_channels(Layout::Ch5_1, 0, 1); // L
        let refs: Vec<&[f32]> = chans.iter().map(|v| v.as_slice()).collect();
        let (l, r) = downmix_to_stereo(Layout::Ch5_1, &refs);
        assert!(l[0] > 0.0 && r[0] == 0.0);

        let chans = impulse_channels(Layout::Ch5_1, 1, 1); // R
        let refs: Vec<&[f32]> = chans.iter().map(|v| v.as_slice()).collect();
        let (l, r) = downmix_to_stereo(Layout::Ch5_1, &refs);
        assert!(r[0] > 0.0 && l[0] == 0.0);
    }

    /// The downmix must be linear: mixing two impulses in different channels
    /// gives the sum of their individual downmixes.
    #[test]
    fn the_downmix_is_linear() {
        let n = 4;
        let mut a = vec![vec![0.0f32; n]; Layout::Ch7_1.channels()];
        a[0][0] = 1.0; // L
        a[4][1] = 0.7; // Ls
        let refs: Vec<&[f32]> = a.iter().map(|v| v.as_slice()).collect();
        let (l, r) = downmix_to_stereo(Layout::Ch7_1, &refs);

        let mut only_l = vec![vec![0.0f32; n]; Layout::Ch7_1.channels()];
        only_l[0][0] = 1.0;
        let refs_l: Vec<&[f32]> = only_l.iter().map(|v| v.as_slice()).collect();
        let (l1, r1) = downmix_to_stereo(Layout::Ch7_1, &refs_l);

        let mut only_ls = vec![vec![0.0f32; n]; Layout::Ch7_1.channels()];
        only_ls[4][1] = 0.7;
        let refs_ls: Vec<&[f32]> = only_ls.iter().map(|v| v.as_slice()).collect();
        let (l2, r2) = downmix_to_stereo(Layout::Ch7_1, &refs_ls);

        for i in 0..n {
            assert!((l[i] - (l1[i] + l2[i])).abs() < 1e-6);
            assert!((r[i] - (r1[i] + r2[i])).abs() < 1e-6);
        }
    }

    #[test]
    #[should_panic(expected = "channel count must match")]
    fn wrong_channel_count_panics() {
        let chans = vec![vec![0.0f32; 4]; 3];
        let refs: Vec<&[f32]> = chans.iter().map(|v| v.as_slice()).collect();
        let _ = downmix_to_stereo(Layout::Ch5_0, &refs);
    }
}
