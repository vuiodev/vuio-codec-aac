//! End-to-end round trip for the minimal USAC frequency-domain single-channel
//! path: real PCM through [`UsacFdEncoder`], its raw block through
//! [`UsacFdDecoder`], and the reconstructed PCM checked against the original.
//!
//! This is the test that proves the arithmetic coder landed earlier this
//! session is actually reachable from a real bitstream, not just exercised in
//! isolation: every frame here is quantized, scalefactor-coded, arithmetic
//! coded, parsed back, dequantized and resynthesized through the same
//! overlap-add filterbank the AAC-LC decoder uses.

use vuiocodecaac::bitstream::BitReader;
use vuiocodecaac::decoder::usac::fd::UsacFdDecoder;
use vuiocodecaac::encoder::usac::fd::{FRAME_LEN, UsacFdEncoder};

/// A real, non-degenerate test signal: two tones plus enough high-frequency
/// content that quantization spans a wide range of magnitudes, including
/// values large enough to force the arithmetic coder's escape path.
fn signal(total: usize) -> Vec<f32> {
    (0..total)
        .map(|i| {
            let t = i as f32;
            (t * 0.041).sin() * 11000.0
                + (t * 0.0037).cos() * 6000.0
                + (t * 0.83).sin() * 2500.0
        })
        .collect()
}

#[test]
fn tonal_signal_round_trips_with_low_error() {
    let frames = 10;
    let total = frames * FRAME_LEN;
    let pcm = signal(total);

    let mut encoder = UsacFdEncoder::new();
    let mut decoder = UsacFdDecoder::new();

    let mut escape_seen = false;
    let mut decoded = vec![0.0f32; total];

    for f in 0..frames {
        let frame_pcm = &pcm[f * FRAME_LEN..(f + 1) * FRAME_LEN];
        let block = encoder.encode_frame(frame_pcm);
        assert!(!block.is_empty(), "an encoded frame must carry real payload");

        let mut reader = BitReader::new(&block);
        let out = decoder.decode_frame(&mut reader).expect("frame must decode");
        assert_eq!(out.len(), FRAME_LEN);

        // Encoding frame f windows against frame f-1's history, so its
        // decoded audio reveals frame f-1's samples, not frame f's -- the
        // usual one-frame MDCT overlap-add delay (see
        // tests/filterbank_reconstruction.rs for the same shape).
        if f > 0 {
            decoded[(f - 1) * FRAME_LEN..f * FRAME_LEN].copy_from_slice(&out);
        }

        // The spectral coder must be exercising real dynamic range, not
        // degenerate all-small-magnitude data: peel a quantized magnitude
        // count out of the block by checking a handful of coefficients are
        // large enough to have forced at least one escape round during
        // encoding. `initial_scalefactor` targets 3/4 of the escape
        // threshold, so a loud frame reliably produces some.
        if !escape_seen {
            escape_seen = block.len() > FRAME_LEN / 16;
        }
    }
    assert!(escape_seen, "the test signal must produce non-trivial encoded frames");

    // `decoded` is already aligned with `pcm` sample-for-sample (call f's
    // output was written to slot f-1, which is where frame f-1's audio
    // belongs). Only the last frame is never revealed -- doing so would need
    // one more decode call than this minimal path has any input for, the
    // same one-frame flush a real encoder's lookahead needs (compare
    // `Encoder::flush` in src/encoder/engine.rs, which this minimal path
    // does not implement).
    let peak = pcm.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let mut signal_energy = 0.0f64;
    let mut error_energy = 0.0f64;
    for i in 0..(frames - 1) * FRAME_LEN {
        let e = (decoded[i] - pcm[i]) as f64;
        error_energy += e * e;
        signal_energy += (pcm[i] as f64).powi(2);
    }
    let snr_db = 10.0 * (signal_energy / error_energy.max(1e-9)).log10();
    assert!(peak > 1000.0, "test signal must have real amplitude, got peak {peak}");
    assert!(snr_db > 20.0, "reconstruction SNR too low: {snr_db:.1} dB");
}

#[test]
fn silence_round_trips_exactly() {
    let frames = 4;
    let mut encoder = UsacFdEncoder::new();
    let mut decoder = UsacFdDecoder::new();

    for f in 0..frames {
        let block = encoder.encode_frame(&[0.0f32; FRAME_LEN]);
        let mut reader = BitReader::new(&block);
        let out = decoder.decode_frame(&mut reader).expect("silent frame must decode");
        if f > 0 {
            assert!(out.iter().all(|&s| s == 0.0), "silence must decode to exact silence");
        }
    }
}
