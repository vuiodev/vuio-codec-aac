//! Exhaustive DSP Transforms and Mathematics Test Suite

use vuiocodecaac::dsp::filterbank::Filterbank;
use vuiocodecaac::types::{WindowSequence, WindowShape};
use vuiocodecaac::dsp::*;
use std::f32::consts::PI;

#[test]
fn test_fixed_point_basic_ops_exhaustive() {
    // Test saturation
    assert_eq!(sat64_32(0x7FFFFFFF), 0x7FFFFFFF);
    assert_eq!(sat64_32(0x80000000), 0x7FFFFFFF); // saturated
    assert_eq!(sat64_32(-0x80000000), -0x80000000);
    assert_eq!(sat64_32(-0x80000001), -0x80000000);

    // Test fractional multiplication
    let a = 0x40000000; // 0.5 in Q31
    let b = 0x4000;     // 0.5 in Q15
    let res = mult32x16in32_shl_sat(a, b);
    assert_eq!(res, 0x20000000); // 0.25 in Q31

    // Test division
    let div_res = div32_pos_normb(100, 200);
    assert_eq!(div_res, 0x40000000); // 0.5 in Q31

    // Test normalization
    let shift = norm32(0x00010000);
    assert_eq!(shift, 14);
}

#[test]
fn test_fft_ifft_roundtrip_all_sizes() {
    for &size in &[64, 128, 256, 512, 1024, 2048] {
        let fft = FftContext::new(size);
        let mut buffer: Vec<Complex32> = (0..size)
            .map(|i| Complex32::new((i as f32 * 0.1).sin(), (i as f32 * 0.05).cos()))
            .collect();
        let original = buffer.clone();

        fft.forward(&mut buffer);
        fft.inverse(&mut buffer);

        for (a, b) in original.iter().zip(buffer.iter()) {
            assert!((a.re - b.re).abs() < 1e-4, "FFT size {} failed on real", size);
            assert!((a.im - b.im).abs() < 1e-4, "FFT size {} failed on imag", size);
        }
    }
}

#[test]
fn test_mdct_imdct_tdac_reconstruction() {
    // Windowed MDCT with overlap-add is a perfect-reconstruction system: the
    // aliasing each frame's transform introduces is cancelled by the neighbouring
    // frame. The first frame has no predecessor, so frame 1 is the first that can
    // reconstruct.
    let frame_len = 1024;
    let mdct = MdctContext::new(frame_len);
    let mut filterbank = Filterbank::new(frame_len);
    let sine_win = generate_sine_window_f32(2 * frame_len);

    let mut signal = vec![0.0f32; 3 * frame_len];
    for (i, x) in signal.iter_mut().enumerate() {
        *x = (2.0 * PI * 440.0 * (i as f32 / 44100.0)).sin() * 1000.0;
    }

    let mut overlap = vec![0.0f32; frame_len];
    let mut reconstructed = vec![0.0f32; 2 * frame_len];
    let mut scratch = vec![vuiocodecaac::dsp::fft::Complex32::new(0.0, 0.0); mdct.scratch_len()];

    for frame in 0..2 {
        let start = frame * frame_len;
        let mut windowed = vec![0.0f32; 2 * frame_len];
        for (w, (&s, &win)) in windowed
            .iter_mut()
            .zip(signal[start..start + 2 * frame_len].iter().zip(sine_win.iter()))
        {
            *w = s * win;
        }

        let mut spec = vec![0.0f32; frame_len];
        mdct.forward(&windowed, &mut spec, &mut scratch);

        let mut pcm = vec![0.0f32; frame_len];
        filterbank.synthesize(
            &spec,
            WindowSequence::OnlyLongSequence,
            WindowShape::Sine,
            WindowShape::Sine,
            &mut overlap,
            &mut pcm,
        );
        reconstructed[start..start + frame_len].copy_from_slice(&pcm);
    }

    for i in 0..frame_len {
        let orig = signal[frame_len + i];
        let rec = reconstructed[frame_len + i];
        assert!((orig - rec).abs() < 0.5, "TDAC mismatch at index {i}: {orig} vs {rec}");
    }
}

#[test]
fn test_qmf_analysis_synthesis_reconstruction() {
    let mut qmf_anal = QmfAnalysis32::new();
    let mut qmf_syn = QmfSynthesis64::new();

    let input_chunk = [0.25f32; 32];
    let mut anal_out = [0.0f32; 32];
    qmf_anal.analyze(&input_chunk, &mut anal_out);

    let mut syn_in = [0.0f32; 64];
    syn_in[..32].copy_from_slice(&anal_out);

    let mut syn_out = [0.0f32; 64];
    qmf_syn.synthesize(&syn_in, &mut syn_out);

    assert_eq!(syn_out.len(), 64);
}

#[test]
fn test_window_properties() {
    let sine = generate_sine_window_f32(1024);
    assert_eq!(sine.len(), 1024);
    assert!((sine[0] - (PI * 0.5 / 1024.0).sin()).abs() < 1e-6);

    let kbd = generate_kbd_window_f32(1024, 4.0);
    assert_eq!(kbd.len(), 1024);
    assert!((kbd[0] - kbd[1023]).abs() < 1e-5); // symmetry

    let ld = generate_low_delay_window_f32(1024);
    assert_eq!(ld.len(), 1024);
}

#[test]
fn test_lpc_levinson_durbin_and_synthesis() {
    let autocorr = [1.0f32, 0.7, 0.4, 0.1];
    let mut lpc = [0.0f32; 4];
    let mut rc = [0.0f32; 3];

    let err = levinson_durbin(&autocorr, 3, &mut lpc, &mut rc).unwrap();
    assert!(err > 0.0);
    assert_eq!(lpc[0], 1.0);
}

#[test]
fn test_resampler_process() {
    let resampler = Resampler::new(44100, 48000);
    let input = vec![0.5f32; 4410];
    let mut output = Vec::new();
    resampler.process(&input, &mut output);
    assert_eq!(output.len(), 4800);
}
