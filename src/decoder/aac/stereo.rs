//! Joint stereo decoding: mid/side and intensity stereo.
//!
//! Both tools let a channel pair share information. M/S transmits sum and difference
//! signals instead of left and right; intensity stereo transmits only the left
//! channel plus a per-band scale from which the right channel is rebuilt. See
//! ISO/IEC 14496-3 clauses 4.6.8.1 and 4.6.8.2.

use crate::decoder::aac::ics::{ChannelData, INTENSITY_HCB, INTENSITY_HCB2};
use crate::tables::scalefactor::{MAX_SFB_LONG, MAX_WINDOWS};

/// Per-band mid/side flags for a channel pair.
#[derive(Debug, Clone)]
pub struct MsMask {
    /// 0: none, 1: per-band flags, 2: all bands.
    pub kind: u8,
    pub used: [[bool; MAX_SFB_LONG]; MAX_WINDOWS],
}

impl Default for MsMask {
    fn default() -> Self {
        Self { kind: 0, used: [[false; MAX_SFB_LONG]; MAX_WINDOWS] }
    }
}

impl MsMask {
    /// Whether M/S applies to band `sfb` of group `g`.
    #[inline]
    pub fn is_used(&self, g: usize, sfb: usize) -> bool {
        match self.kind {
            1 => self.used[g][sfb],
            2 => true,
            _ => false,
        }
    }
}

/// Undo mid/side coding in place over a channel pair.
///
/// The left channel carries mid and the right carries side; recovering left and
/// right is `l = m + s`, `r = m - s`. Bands coded as intensity or noise are skipped
/// because their contents are produced by the later tools.
pub fn apply_ms_stereo(left: &mut ChannelData, right: &mut ChannelData, mask: &MsMask) {
    if mask.kind == 0 {
        return;
    }
    let ics = left.ics.clone();

    for g in 0..ics.num_window_groups {
        let group_base = ics.group_base(g);
        let group_len = ics.window_group_length[g];
        for sfb in 0..ics.max_sfb {
            if !mask.is_used(g, sfb) {
                continue;
            }
            // Intensity bands are reconstructed from the left channel later, and
            // noise bands are generated independently; neither is a valid M/S pair.
            let cb = right.sfb_cb[g][sfb];
            if cb == INTENSITY_HCB || cb == INTENSITY_HCB2 {
                continue;
            }

            let start = group_base + ics.grouped_offset(g, sfb);
            let width = (ics.swb_offset[sfb + 1] - ics.swb_offset[sfb]) as usize * group_len;
            let end = (start + width).min(left.spec.len()).min(right.spec.len());

            for i in start..end {
                let m = left.spec[i];
                let s = right.spec[i];
                left.spec[i] = m + s;
                right.spec[i] = m - s;
            }
        }
    }
}

/// Rebuild intensity-stereo bands of the right channel from the left.
///
/// The transmitted scalefactor is an intensity *position*: the right channel is the
/// left scaled by `2^(-position/4)`, with the sign chosen by which of the two
/// intensity codebooks was used and flipped again when M/S is active on that band.
pub fn apply_intensity_stereo(left: &ChannelData, right: &mut ChannelData, mask: &MsMask) {
    let ics = right.ics.clone();

    for g in 0..ics.num_window_groups {
        let group_base = ics.group_base(g);
        let group_len = ics.window_group_length[g];
        for sfb in 0..ics.max_sfb {
            let phase = match right.sfb_cb[g][sfb] {
                INTENSITY_HCB => 1.0f32,
                INTENSITY_HCB2 => -1.0f32,
                _ => continue,
            };
            // M/S on an intensity band inverts the reconstructed phase.
            let sign = if mask.is_used(g, sfb) { -phase } else { phase };

            let position = right.scale_factors[g][sfb] as f32;
            let scale = sign * (-0.25 * position).exp2();

            let start = group_base + ics.grouped_offset(g, sfb);
            let width = (ics.swb_offset[sfb + 1] - ics.swb_offset[sfb]) as usize * group_len;
            let end = (start + width).min(left.spec.len()).min(right.spec.len());

            for i in start..end {
                right.spec[i] = left.spec[i] * scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::aac::ics::IcsInfo;
    use crate::tables::scalefactor::compute_sfb_offsets;

    fn long_channel(n: usize) -> ChannelData {
        let mut ch = ChannelData::new(n);
        let widths = crate::tables::sfb::SFB_48_1024;
        let mut offsets = [0usize; MAX_SFB_LONG + 1];
        let count = compute_sfb_offsets(widths, &mut offsets);
        let mut ics = IcsInfo { window_length: n, ..Default::default() };
        ics.num_swb = count - 1;
        ics.max_sfb = ics.num_swb;
        for (d, &s) in ics.swb_offset.iter_mut().zip(offsets.iter()) {
            *d = s as u16;
        }
        ch.ics = ics;
        ch
    }

    /// M/S is an involution up to scale: applying it to (m, s) gives (m+s, m-s), and
    /// applying it again gives (2m, 2s).
    #[test]
    fn ms_stereo_applied_twice_doubles() {
        let mut l = long_channel(1024);
        let mut r = long_channel(1024);
        for i in 0..1024 {
            l.spec[i] = (i as f32) * 0.5;
            r.spec[i] = 100.0 - i as f32;
        }
        let original_l = l.spec.clone();
        let original_r = r.spec.clone();

        let mask = MsMask { kind: 2, ..Default::default() };
        apply_ms_stereo(&mut l, &mut r, &mask);
        apply_ms_stereo(&mut l, &mut r, &mask);

        for i in 0..1024 {
            assert!((l.spec[i] - 2.0 * original_l[i]).abs() < 1e-3, "left {i}");
            assert!((r.spec[i] - 2.0 * original_r[i]).abs() < 1e-3, "right {i}");
        }
    }

    /// With the mask off, M/S must not touch anything.
    #[test]
    fn ms_stereo_respects_an_empty_mask() {
        let mut l = long_channel(1024);
        let mut r = long_channel(1024);
        l.spec.fill(3.0);
        r.spec.fill(-7.0);
        apply_ms_stereo(&mut l, &mut r, &MsMask::default());
        assert!(l.spec.iter().all(|&v| v == 3.0));
        assert!(r.spec.iter().all(|&v| v == -7.0));
    }

    /// Per-band flags must apply to exactly the flagged bands.
    #[test]
    fn ms_stereo_honours_per_band_flags() {
        let mut l = long_channel(1024);
        let mut r = long_channel(1024);
        l.spec.fill(10.0);
        r.spec.fill(4.0);

        let mut mask = MsMask { kind: 1, ..Default::default() };
        mask.used[0][3] = true;

        let lo = l.ics.swb_offset[3] as usize;
        let hi = l.ics.swb_offset[4] as usize;
        apply_ms_stereo(&mut l, &mut r, &mask);

        for i in lo..hi {
            assert_eq!(l.spec[i], 14.0, "band 3 line {i} should be m+s");
            assert_eq!(r.spec[i], 6.0);
        }
        assert_eq!(l.spec[0], 10.0, "band 0 must be untouched");
        assert_eq!(r.spec[0], 4.0);
    }

    /// An intensity position of zero copies the left channel; each step of four
    /// halves the amplitude.
    #[test]
    fn intensity_scale_follows_the_position() {
        for (position, expect) in [(0i16, 1.0f32), (4, 0.5), (8, 0.25), (-4, 2.0)] {
            let mut l = long_channel(1024);
            let mut r = long_channel(1024);
            l.spec.fill(8.0);
            r.sfb_cb[0][2] = INTENSITY_HCB;
            r.scale_factors[0][2] = position;

            apply_intensity_stereo(&l, &mut r, &MsMask::default());

            let lo = r.ics.swb_offset[2] as usize;
            assert!(
                (r.spec[lo] - 8.0 * expect).abs() < 1e-4,
                "position {position}: got {} want {}",
                r.spec[lo],
                8.0 * expect
            );
        }
    }

    /// Codebook 14 is the out-of-phase variant, and M/S flips the phase again.
    #[test]
    fn intensity_phase_follows_codebook_and_ms() {
        let cases = [
            (INTENSITY_HCB, 0u8, 1.0f32),
            (INTENSITY_HCB2, 0, -1.0),
            (INTENSITY_HCB, 2, -1.0),
            (INTENSITY_HCB2, 2, 1.0),
        ];
        for (cb, ms_kind, expect) in cases {
            let mut l = long_channel(1024);
            let mut r = long_channel(1024);
            l.spec.fill(5.0);
            r.sfb_cb[0][1] = cb;
            r.scale_factors[0][1] = 0;

            let mask = MsMask { kind: ms_kind, ..Default::default() };
            apply_intensity_stereo(&l, &mut r, &mask);

            let lo = r.ics.swb_offset[1] as usize;
            assert!(
                (r.spec[lo] - 5.0 * expect).abs() < 1e-4,
                "cb {cb} ms {ms_kind}: got {} want {}",
                r.spec[lo],
                5.0 * expect
            );
        }
    }

    /// M/S must leave intensity bands alone, since intensity overwrites them.
    #[test]
    fn ms_skips_intensity_bands() {
        let mut l = long_channel(1024);
        let mut r = long_channel(1024);
        l.spec.fill(10.0);
        r.spec.fill(4.0);
        r.sfb_cb[0][5] = INTENSITY_HCB;

        let mask = MsMask { kind: 2, ..Default::default() };
        apply_ms_stereo(&mut l, &mut r, &mask);

        let lo = l.ics.swb_offset[5] as usize;
        assert_eq!(l.spec[lo], 10.0, "intensity band must not be M/S decoded");
        assert_eq!(r.spec[lo], 4.0);
    }
}
