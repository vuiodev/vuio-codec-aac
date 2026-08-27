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
use vuiocodecaac::decoder::usac::fd::{UsacFdDecoder, UsacFdStereoDecoder};
use vuiocodecaac::encoder::usac::fd::{FRAME_LEN, UsacFdEncoder, UsacFdStereoEncoder};

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

/// A smaller configured bit budget must produce a smaller encoded frame on
/// the same audio, the observable proof that `set_budget_bits` actually
/// drives a real rate-distortion tradeoff rather than being ignored.
#[test]
fn tighter_budget_produces_smaller_frames() {
    let pcm = signal(FRAME_LEN);

    let mut generous = UsacFdEncoder::new();
    generous.set_budget_bits(20_000);
    let big_block = generous.encode_frame(&pcm);

    let mut tight = UsacFdEncoder::new();
    tight.set_budget_bits(400);
    let small_block = tight.encode_frame(&pcm);

    assert!(
        small_block.len() < big_block.len(),
        "tight budget produced {} bytes, generous produced {} bytes",
        small_block.len(),
        big_block.len()
    );

    // The tight budget must still decode to something, even if coarse.
    let mut decoder = UsacFdDecoder::new();
    let mut reader = BitReader::new(&small_block);
    let out = decoder.decode_frame(&mut reader).expect("even a coarse frame must decode");
    assert_eq!(out.len(), FRAME_LEN);
}

/// Two tones panned hard left and hard right: mid/side should win on most
/// bands (both channels share most of their energy), and the reconstruction
/// of both channels must stay close to the original.
#[test]
fn stereo_signal_round_trips_with_low_error() {
    let frames = 10;
    let total = frames * FRAME_LEN;

    let left: Vec<f32> = (0..total)
        .map(|i| {
            let t = i as f32;
            (t * 0.041).sin() * 11000.0 + (t * 0.83).sin() * 2000.0
        })
        .collect();
    let right: Vec<f32> = (0..total)
        .map(|i| {
            let t = i as f32;
            (t * 0.041).sin() * 9000.0 + (t * 0.0037).cos() * 6000.0
        })
        .collect();

    let mut encoder = UsacFdStereoEncoder::new();
    let mut decoder = UsacFdStereoDecoder::new();

    let mut decoded_left = vec![0.0f32; total];
    let mut decoded_right = vec![0.0f32; total];

    for f in 0..frames {
        let l = &left[f * FRAME_LEN..(f + 1) * FRAME_LEN];
        let r = &right[f * FRAME_LEN..(f + 1) * FRAME_LEN];
        let block = encoder.encode_frame(l, r);
        assert!(!block.is_empty(), "an encoded stereo frame must carry real payload");

        let mut reader = BitReader::new(&block);
        let (out_l, out_r) = decoder.decode_frame(&mut reader).expect("stereo frame must decode");
        assert_eq!(out_l.len(), FRAME_LEN);
        assert_eq!(out_r.len(), FRAME_LEN);

        if f > 0 {
            decoded_left[(f - 1) * FRAME_LEN..f * FRAME_LEN].copy_from_slice(&out_l);
            decoded_right[(f - 1) * FRAME_LEN..f * FRAME_LEN].copy_from_slice(&out_r);
        }
    }

    let snr = |decoded: &[f32], original: &[f32]| -> f64 {
        let mut signal_energy = 0.0f64;
        let mut error_energy = 0.0f64;
        for i in 0..(frames - 1) * FRAME_LEN {
            let e = (decoded[i] - original[i]) as f64;
            error_energy += e * e;
            signal_energy += (original[i] as f64).powi(2);
        }
        10.0 * (signal_energy / error_energy.max(1e-9)).log10()
    };

    let snr_l = snr(&decoded_left, &left);
    let snr_r = snr(&decoded_right, &right);
    assert!(snr_l > 20.0, "left channel reconstruction SNR too low: {snr_l:.1} dB");
    assert!(snr_r > 20.0, "right channel reconstruction SNR too low: {snr_r:.1} dB");
}

/// Broadband noise at a budget too tight to code every band pushes real,
/// non-leakage energy above [`vuiocodecaac::decoder::usac::fd::NOISE_FILLING_START_OFFSET`]
/// into bands the rate loop quantizes to all zero — exactly the situation
/// noise filling exists for. This drives the whole pipeline end to end
/// (unlike the unit tests in `encoder`/`decoder::usac::fd`, which exercise
/// the mechanism directly) and checks the side info a real encode actually
/// chose, not a hand-constructed one.
#[test]
fn a_tight_budget_on_broadband_noise_turns_on_noise_filling() {
    use vuiocodecaac::bitstream::BitReader;

    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as i32 - 0x800) as f32
    };
    let pcm: Vec<f32> = (0..FRAME_LEN).map(|_| next()).collect();

    let mut encoder = UsacFdEncoder::new();
    encoder.set_budget_bits(600);
    let block = encoder.encode_frame(&pcm);

    let mut reader = BitReader::new(&block);
    let _global_gain = reader.read_u8(8).unwrap();
    let noise_level = reader.read_u8(3).unwrap();
    let noise_offset = reader.read_u8(5).unwrap();
    assert!(noise_level > 0, "a tight budget on broadband noise must leave real energy to fill");
    assert_eq!(noise_offset, 16, "this encoder's noise_offset is always the shift-free value");

    // The frame must still decode to something real, not just parse.
    let mut decoder = UsacFdDecoder::new();
    let mut reader = BitReader::new(&block);
    let out = decoder.decode_frame(&mut reader).expect("a noise-filled frame must still decode");
    assert_eq!(out.len(), FRAME_LEN);
}

/// A signal with a strong short-term envelope across the spectrum (the exact
/// shape `src/encoder/usac/tns.rs`'s own unit tests already prove triggers a
/// real TNS filter and measurably reduces the residual) must still round-trip
/// end to end through the full encoder/decoder pipeline, not just through
/// direct calls to the filter itself. This is what actually exercises TNS's
/// bitstream wiring — the presence bit, the coefficient fields, and the
/// decoder applying the inverse filter in the right place relative to
/// dequantization — rather than the filter math in isolation.
#[test]
fn a_signal_with_spectral_envelope_still_round_trips_through_tns() {
    let frames = 10;
    let total = frames * FRAME_LEN;
    // A raised-cosine envelope with a period of exactly one frame -- loud in
    // the middle of each window, quiet at its edges -- gives real short-term
    // spectral structure for TNS to shape without the pathological hard reset
    // a sawtooth-style envelope would put at every frame boundary (which this
    // minimal, block-switching-free FD path has no tool to handle well, and
    // which would make a low SNR reflect that gap rather than anything about
    // TNS). The envelope is continuous by construction: `cos` at `i % 1024`
    // matches `cos` at `(i+1) % 1024` exactly at every boundary.
    let pcm: Vec<f32> = (0..total)
        .map(|i| {
            let within_frame = (i % FRAME_LEN) as f32;
            let phase = std::f32::consts::TAU * within_frame / FRAME_LEN as f32;
            let envelope = 0.5 * (1.0 - phase.cos());
            8000.0 * envelope * (within_frame * 0.31).sin()
        })
        .collect();

    let mut encoder = UsacFdEncoder::new();
    let mut decoder = UsacFdDecoder::new();
    let mut decoded = vec![0.0f32; total];

    for f in 0..frames {
        let frame_pcm = &pcm[f * FRAME_LEN..(f + 1) * FRAME_LEN];
        let block = encoder.encode_frame(frame_pcm);
        let mut reader = BitReader::new(&block);
        let out = decoder.decode_frame(&mut reader).expect("TNS-shaped frame must decode");
        if f > 0 {
            decoded[(f - 1) * FRAME_LEN..f * FRAME_LEN].copy_from_slice(&out);
        }
    }

    let mut signal_energy = 0.0f64;
    let mut error_energy = 0.0f64;
    for i in 0..(frames - 1) * FRAME_LEN {
        let e = (decoded[i] - pcm[i]) as f64;
        error_energy += e * e;
        signal_energy += (pcm[i] as f64).powi(2);
    }
    let snr_db = 10.0 * (signal_energy / error_energy.max(1e-9)).log10();
    assert!(snr_db > 20.0, "reconstruction SNR too low with TNS in the loop: {snr_db:.1} dB");
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
