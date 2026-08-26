//! Joint Stereo Processing (Mid/Side and Intensity Stereo)
//!
//! Applies Mid/Side (M/S) matrix decoding ($L = M + S, R = M - S$)
//! and Intensity Stereo reconstruction across active scalefactor bands.

/// Apply Mid/Side (M/S) stereo matrix to left and right channel spectra.
pub fn apply_ms_stereo(left: &mut [f32], right: &mut [f32]) {
    assert_eq!(left.len(), right.len());
    for (l, r) in left.iter_mut().zip(right.iter_mut()) {
        let m = *l;
        let s = *r;
        *l = m + s;
        *r = m - s;
    }
}

/// Apply Intensity Stereo reconstruction to target channel based on source channel and scalefactor.
pub fn apply_intensity_stereo(
    source: &[f32],
    is_scale: i16,
    invert: bool,
    target: &mut [f32],
) {
    assert_eq!(source.len(), target.len());
    let scale = 0.5f32.powf(is_scale as f32 * 0.25);
    let scale_factor = if invert { -scale } else { scale };

    for (s, t) in source.iter().zip(target.iter_mut()) {
        *t = *s * scale_factor;
    }
}
