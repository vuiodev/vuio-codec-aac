//! End-to-end perfect-reconstruction test for the synthesis filterbank.
//!
//! Runs a signal through the analysis side (windowed forward MDCT, exactly as an
//! encoder would) and back through [`Filterbank`], across every window-sequence
//! transition AAC allows. Windowed MDCT with overlap-add is a perfect-reconstruction
//! system, so any mismatch in a window shape, a transition region, or the placement
//! of the eight short transforms shows up here as reconstruction error.

use vuiocodecaac::dsp::filterbank::{Filterbank, frame_window, short_window, short_window_offset};
use vuiocodecaac::types::{WindowSequence, WindowShape};

/// Forward MDCT from the definition: `2m` windowed samples to `m` coefficients.
///
/// Carries the factor of 2 the standard's analysis transform specifies, which pairs
/// with the `1/m` the synthesis side applies.
fn forward_mdct(input: &[f32], m: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m];
    for (k, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for (i, &x) in input.iter().enumerate() {
            let a = std::f64::consts::PI / m as f64
                * (i as f64 + 0.5 + m as f64 / 2.0)
                * (k as f64 + 0.5);
            acc += x as f64 * a.cos();
        }
        *o = (acc * 2.0) as f32;
    }
    out
}

/// Produce the spectral coefficients an encoder would emit for one frame.
///
/// `block` is the `2n` input samples this frame covers.
fn analyze(
    block: &[f32],
    n: usize,
    sequence: WindowSequence,
    shape: WindowShape,
    prev_shape: WindowShape,
) -> Vec<f32> {
    if sequence == WindowSequence::EightShortSequence {
        let sn = n / 8;
        let mut spec = vec![0.0f32; n];
        for w in 0..8 {
            let win = short_window(n, shape, prev_shape, w);
            let off = short_window_offset(n, w);
            let mut sub = vec![0.0f32; 2 * sn];
            for i in 0..2 * sn {
                sub[i] = block[off + i] * win[i];
            }
            spec[w * sn..(w + 1) * sn].copy_from_slice(&forward_mdct(&sub, sn));
        }
        spec
    } else {
        let win = frame_window(n, sequence, shape, prev_shape);
        let windowed: Vec<f32> = block.iter().zip(win.iter()).map(|(x, w)| x * w).collect();
        forward_mdct(&windowed, n)
    }
}

/// Round-trip `sequences` through analysis and synthesis, returning the worst
/// reconstruction error over the frames where overlap-add has fully primed.
fn round_trip_error(n: usize, sequences: &[(WindowSequence, WindowShape)]) -> f32 {
    let frames = sequences.len();
    let total = frames * n;

    let signal: Vec<f32> = (0..total + 2 * n)
        .map(|i| {
            let t = i as f32;
            (t * 0.021).sin() * 6000.0 + (t * 0.0037).cos() * 2500.0 + (t * 0.31).sin() * 800.0
        })
        .collect();

    let mut fb = Filterbank::new(n);
    let mut overlap = vec![0.0f32; n];
    let mut recon = vec![0.0f32; total];
    let mut prev_shape = WindowShape::Sine;

    for (f, &(sequence, shape)) in sequences.iter().enumerate() {
        let start = f * n;
        let spec = analyze(&signal[start..start + 2 * n], n, sequence, shape, prev_shape);

        let mut out = vec![0.0f32; n];
        fb.synthesize(&spec, sequence, shape, prev_shape, &mut overlap, &mut out);
        recon[start..start + n].copy_from_slice(&out);
        prev_shape = shape;
    }

    // The first frame has no prior overlap, so it can never reconstruct.
    let mut worst = 0.0f32;
    for i in n..total {
        let e = (recon[i] - signal[i]).abs();
        if e > worst {
            worst = e;
        }
    }
    worst
}

/// A run of long frames must reconstruct, for both window shapes.
#[test]
fn long_sequences_reconstruct() {
    for shape in [WindowShape::Sine, WindowShape::Kbd] {
        let seqs = vec![(WindowSequence::OnlyLongSequence, shape); 6];
        let err = round_trip_error(256, &seqs);
        assert!(err < 2.0, "{shape:?} long run: worst error {err}");
    }
}

/// The full block-switching cycle must reconstruct: long, start, eight short, stop,
/// long. This is the transition the decoder gets wrong if the short transforms are
/// misplaced or the transition windows are malformed.
#[test]
fn block_switch_cycle_reconstructs() {
    for shape in [WindowShape::Sine, WindowShape::Kbd] {
        let seqs = vec![
            (WindowSequence::OnlyLongSequence, shape),
            (WindowSequence::OnlyLongSequence, shape),
            (WindowSequence::LongStartSequence, shape),
            (WindowSequence::EightShortSequence, shape),
            (WindowSequence::LongStopSequence, shape),
            (WindowSequence::OnlyLongSequence, shape),
            (WindowSequence::OnlyLongSequence, shape),
        ];
        let err = round_trip_error(256, &seqs);
        assert!(err < 2.0, "{shape:?} block switch cycle: worst error {err}");
    }
}

/// Consecutive short frames (start, short, short, stop) must also reconstruct.
#[test]
fn consecutive_short_frames_reconstruct() {
    let shape = WindowShape::Sine;
    let seqs = vec![
        (WindowSequence::OnlyLongSequence, shape),
        (WindowSequence::LongStartSequence, shape),
        (WindowSequence::EightShortSequence, shape),
        (WindowSequence::EightShortSequence, shape),
        (WindowSequence::EightShortSequence, shape),
        (WindowSequence::LongStopSequence, shape),
        (WindowSequence::OnlyLongSequence, shape),
    ];
    let err = round_trip_error(256, &seqs);
    assert!(err < 2.0, "consecutive short frames: worst error {err}");
}

/// A shape change across a frame boundary must still reconstruct, since each frame
/// takes its rising edge from the previous frame's shape.
#[test]
fn window_shape_changes_reconstruct() {
    let seqs = vec![
        (WindowSequence::OnlyLongSequence, WindowShape::Sine),
        (WindowSequence::OnlyLongSequence, WindowShape::Kbd),
        (WindowSequence::OnlyLongSequence, WindowShape::Kbd),
        (WindowSequence::OnlyLongSequence, WindowShape::Sine),
        (WindowSequence::LongStartSequence, WindowShape::Kbd),
        (WindowSequence::EightShortSequence, WindowShape::Sine),
        (WindowSequence::LongStopSequence, WindowShape::Kbd),
        (WindowSequence::OnlyLongSequence, WindowShape::Sine),
    ];
    let err = round_trip_error(256, &seqs);
    assert!(err < 2.0, "shape changes: worst error {err}");
}

/// The same cycle at the production transform size.
#[test]
fn block_switch_cycle_reconstructs_at_1024() {
    let seqs = vec![
        (WindowSequence::OnlyLongSequence, WindowShape::Kbd),
        (WindowSequence::LongStartSequence, WindowShape::Kbd),
        (WindowSequence::EightShortSequence, WindowShape::Kbd),
        (WindowSequence::LongStopSequence, WindowShape::Kbd),
        (WindowSequence::OnlyLongSequence, WindowShape::Kbd),
    ];
    let err = round_trip_error(1024, &seqs);
    assert!(err < 4.0, "1024-point block switch cycle: worst error {err}");
}
